from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, HttpResponse, redirect, get_object_or_404
from django.contrib import messages
import bcrypt
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .serializers import FileSerializer
from django.contrib.auth import logout
from .models import User, AuthorizedPerson, UnauthorizedDetection, File
from django.contrib.auth import authenticate, login as auth_login
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta
from django.db.models.functions import TruncDate

from django.contrib.auth import get_user_model

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, StreamingHttpResponse
import threading
import time
import requests
import uuid
from django.core.files.base import ContentFile
from django.conf import settings
from .models import SMSConfiguration, AlertRecipient

# Arkesel SMS Gateway Integration
class ArkeselSMSService:
    def __init__(self):
        self.base_url = "https://sms.arkesel.com/api/v2/sms"
        self.config = self.get_active_config()
    
    def get_active_config(self):
        """Get active SMS configuration"""
        try:
            return SMSConfiguration.objects.filter(is_active=True).first()
        except:
            return None
    
    def send_unauthorized_alert(self, detection, captured_image_url=None):
        """Send SMS alert for unauthorized access detection"""
        if not self.config or not self.config.send_alerts_enabled:
            return {'success': False, 'error': 'SMS alerts disabled or not configured'}

        # Get recipients for unauthorized access alerts
        recipients = self.get_alert_recipients()
        if not recipients:
            return {'success': False, 'error': 'No active recipients found'}

        # Create alert message
        message = self.create_alert_message(detection, captured_image_url)

        # Send SMS to all recipients
        results = []
        for recipient in recipients:
            result = self.send_sms(recipient.phone_number, message)
            results.append({
                'recipient': recipient.name,
                'phone': recipient.phone_number,
                'success': result['success'],
                'message_id': result.get('message_id'),
                'error': result.get('error')
            })

        return {
            'success': True,
            'total_sent': len([r for r in results if r['success']]),
            'total_failed': len([r for r in results if not r['success']]),
            'results': results
        }
    
    def get_alert_recipients(self):
        """Get recipients for unauthorized access alerts"""
        recipients = AlertRecipient.objects.filter(is_active=True, alert_for_unauthorized=True)
        return recipients.order_by('priority_level')
    
    def create_alert_message(self, detection, captured_image_url=None):
        """Create SMS alert message for unauthorized access"""
        timestamp = detection.timestamp.strftime('%Y-%m-%d %H:%M:%S')

        message = f"🚨 UNAUTHORIZED ACCESS ALERT 🚨\n"
        name = detection.detected_name or "Unknown Person"
        message += f"Person: {name}\n"
        message += f"Area: {detection.access_attempted}\n"
        message += f"Time: {timestamp}\n"

        if self.config.include_location:
            if detection.camera_location:
                message += f"Location: {detection.camera_location}\n"
            if detection.latitude and detection.longitude:
                message += f"Coordinates: {detection.latitude}, {detection.longitude}\n"
                if hasattr(detection, 'google_maps_link'):
                    message += f"Map: {detection.google_maps_link}\n"

        if captured_image_url and self.config.include_image_link:
            message += f"Image: {captured_image_url}\n"

        message += f"Confidence: {detection.detection_confidence:.2f}\n" if detection.detection_confidence else ""
        message += "Security team notified. Please investigate immediately."

        return message
    
    def send_sms(self, phone_number, message):
        """Send individual SMS via Arkesel API"""
        if not self.config:
            return {'success': False, 'error': 'SMS not configured'}
        
        headers = {
            'api-key': self.config.api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'sender': self.config.sender_id,
            'message': message,
            'recipients': [phone_number]
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            response_data = response.json()
            
            if response.status_code == 200 and response_data.get('status') == 'success':
                return {
                    'success': True,
                    'message_id': response_data.get('data', {}).get('message_id'),
                    'response': response_data
                }
            else:
                return {
                    'success': False,
                    'error': response_data.get('message', 'SMS sending failed'),
                    'response': response_data
                }
        
        except requests.RequestException as e:
            return {
                'success': False,
                'error': f'Network error: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'SMS error: {str(e)}'
            }
    
    def check_balance(self):
        """Check SMS balance via Arkesel API"""
        if not self.config:
            return {'success': False, 'error': 'SMS not configured'}
        
        headers = {'api-key': self.config.api_key}
        
        try:
            response = requests.get(f"{self.base_url.replace('sms', 'clients')}/balance-details", 
                                  headers=headers, timeout=30)
            response_data = response.json()
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'balance': response_data.get('data', {}).get('balance', 0),
                    'response': response_data
                }
            else:
                return {
                    'success': False,
                    'error': response_data.get('message', 'Balance check failed')
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Balance check error: {str(e)}'
            }

def save_captured_image(frame, person_name, detection_id):
    """Save captured frame as evidence image for unauthorized detection"""
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"unauthorized_{person_name.replace(' ', '_')}_{timestamp}_{detection_id}.jpg"

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ret:
            image_content = ContentFile(buffer.tobytes(), name=filename)
            return image_content
    except Exception as e:
        print(f"❌ Error saving captured image: {e}")

    return None

def get_camera_location(latitude=None, longitude=None):
    """
    Get camera location with GPS coordinates
    Uses provided coordinates or defaults to campus location
    """
    if latitude and longitude:
        # Use provided GPS coordinates
        return {
            'description': f'Security Camera at {latitude}, {longitude}',
            'latitude': latitude,
            'longitude': longitude,
            'address': f'GPS Location: {latitude}, {longitude}',
            'area_name': f'Area near {latitude}, {longitude}'
        }

    # Default campus location (can be enhanced with actual GPS from device)
    return {
        'description': 'Security Camera - University Campus',
        'latitude': 5.6037,  # Default Accra coordinates
        'longitude': -0.1870,
        'address': 'University Campus, Accra, Ghana',
        'area_name': 'Main Campus'
    }

from datetime import datetime
import json
import base64
from skimage import exposure, filters
from PIL import ImageEnhance

def process_authorized_face(image_path, full_name):
    """
    Advanced face processing function that extracts multiple embeddings
    and applies various image enhancement techniques for better accuracy
    """
    processing_notes = []
    face_embeddings = []
    face_embeddings_backup = []
    face_detected = False
    face_detection_confidence = 0.0
    face_quality_score = 0.0
    primary_embedding = None
    
    try:
        # Load original image
        original_img = cv2.imread(image_path)
        if original_img is None:
            return {
                'face_detected': False,
                'face_embedding': None,
                'face_embeddings_backup': None,
                'face_quality_score': 0.0,
                'face_detection_confidence': 0.0,
                'processing_notes': 'Failed to load image file',
                'error': 'Could not read the uploaded image file.'
            }
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        processing_notes.append(f"Original image size: {img_rgb.shape}")
        
        # Try face detection on original image
        faces = face_app.get(img_rgb)
        
        if len(faces) > 0:
            face_detected = True
            face_detection_confidence = float(faces[0].det_score)
            primary_face = faces[0]
            primary_embedding = primary_face.normed_embedding.tolist()
            face_embeddings.append(primary_embedding)
            processing_notes.append(f"Primary face detected with confidence: {face_detection_confidence:.3f}")
            
            # Calculate initial quality score based on face size and detection confidence
            bbox = primary_face.bbox
            face_width = bbox[2] - bbox[0]
            face_height = bbox[3] - bbox[1]
            face_area = face_width * face_height
            image_area = img_rgb.shape[0] * img_rgb.shape[1]
            face_size_ratio = face_area / image_area
            
            # Quality score combines detection confidence and face size
            base_quality = face_detection_confidence * min(1.0, face_size_ratio * 10)
            processing_notes.append(f"Face size ratio: {face_size_ratio:.4f}, Base quality: {base_quality:.3f}")
            
        else:
            processing_notes.append("No face detected in original image")
        
        # Apply image enhancement techniques to get more embeddings
        enhanced_images = []
        
        # Enhancement 1: Histogram equalization
        try:
            img_yuv = cv2.cvtColor(original_img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            enhanced_images.append(('histogram_equalized', img_eq))
        except Exception as e:
            processing_notes.append(f"Histogram equalization failed: {str(e)}")
        
        # Enhancement 2: Brightness and contrast adjustment
        try:
            pil_img = Image.fromarray(img_rgb)
            enhancer = ImageEnhance.Brightness(pil_img)
            bright_img = enhancer.enhance(1.2)  # Increase brightness by 20%
            enhancer = ImageEnhance.Contrast(bright_img)
            contrast_img = enhancer.enhance(1.3)  # Increase contrast by 30%
            enhanced_images.append(('brightness_contrast', np.array(contrast_img)))
        except Exception as e:
            processing_notes.append(f"Brightness/contrast enhancement failed: {str(e)}")
        
        # Enhancement 3: Gamma correction
        try:
            gamma_corrected = exposure.adjust_gamma(img_rgb, gamma=0.8)
            gamma_corrected = (gamma_corrected * 255).astype(np.uint8)
            enhanced_images.append(('gamma_corrected', gamma_corrected))
        except Exception as e:
            processing_notes.append(f"Gamma correction failed: {str(e)}")
        
        # Enhancement 4: Sharpening
        try:
            # Convert to grayscale for sharpening, then back to RGB
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            sharpened_gray = filters.unsharp_mask(gray, radius=1, amount=1)
            sharpened_gray = (sharpened_gray * 255).astype(np.uint8)
            sharpened_rgb = cv2.cvtColor(sharpened_gray, cv2.COLOR_GRAY2RGB)
            enhanced_images.append(('sharpened', sharpened_rgb))
        except Exception as e:
            processing_notes.append(f"Sharpening failed: {str(e)}")
        
        # Extract faces from enhanced images
        quality_scores = []
        if face_detected:
            quality_scores.append(base_quality)
        
        for enhancement_name, enhanced_img in enhanced_images:
            try:
                faces_enhanced = face_app.get(enhanced_img)
                if len(faces_enhanced) > 0:
                    best_face = faces_enhanced[0]
                    embedding = best_face.normed_embedding.tolist()
                    face_embeddings_backup.append({
                        'embedding': embedding,
                        'enhancement': enhancement_name,
                        'confidence': float(best_face.det_score)
                    })
                    
                    # Calculate quality for this enhancement
                    bbox = best_face.bbox
                    face_width = bbox[2] - bbox[0]
                    face_height = bbox[3] - bbox[1]
                    face_area = face_width * face_height
                    image_area = enhanced_img.shape[0] * enhanced_img.shape[1]
                    face_size_ratio = face_area / image_area
                    enhancement_quality = best_face.det_score * min(1.0, face_size_ratio * 10)
                    quality_scores.append(enhancement_quality)
                    
                    processing_notes.append(f"Face detected in {enhancement_name} with confidence: {best_face.det_score:.3f}")
                    
                    # If this is better than our primary embedding, use it as primary
                    if not face_detected or enhancement_quality > face_quality_score:
                        face_detected = True
                        face_detection_confidence = float(best_face.det_score)
                        primary_embedding = embedding
                        face_quality_score = enhancement_quality
                        processing_notes.append(f"Using {enhancement_name} as primary embedding")
                        
            except Exception as e:
                processing_notes.append(f"Enhancement {enhancement_name} processing failed: {str(e)}")
        
        # Calculate final quality score
        if quality_scores:
            face_quality_score = max(quality_scores)  # Use the best quality score
            avg_quality = np.mean(quality_scores)
            processing_notes.append(f"Final quality score: {face_quality_score:.3f} (avg: {avg_quality:.3f})")
        
        # Additional quality factors
        if face_detected and len(face_embeddings_backup) > 0:
            # Bonus for having multiple good embeddings
            face_quality_score = min(1.0, face_quality_score + 0.1)
            processing_notes.append(f"Quality bonus for multiple embeddings: {face_quality_score:.3f}")
        
        processing_notes.append(f"Total embeddings generated: {len(face_embeddings_backup) + (1 if primary_embedding else 0)}")
        
        return {
            'face_detected': face_detected,
            'face_embedding': primary_embedding,
            'face_embeddings_backup': face_embeddings_backup,
            'face_quality_score': face_quality_score,
            'face_detection_confidence': face_detection_confidence,
            'processing_notes': ' | '.join(processing_notes),
            'error': None
        }
        
    except Exception as e:
        error_msg = f"Face processing failed for {full_name}: {str(e)}"
        processing_notes.append(error_msg)
        return {
            'face_detected': False,
            'face_embedding': None,
            'face_embeddings_backup': None,
            'face_quality_score': 0.0,
            'face_detection_confidence': 0.0,
            'processing_notes': ' | '.join(processing_notes),
            'error': error_msg
        }





User = get_user_model()

# Initialize InsightFace model (detection + recognition)
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

class FileView(APIView):
  parser_classes = (MultiPartParser, FormParser)
  def post(self, request, *args, **kwargs):
    file_serializer = FileSerializer(data=request.data)
    if file_serializer.is_valid():
      file_serializer.save()
      return Response(file_serializer.data, status=status.HTTP_201_CREATED)
    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, email=email, password=password)
        
        if user is not None:
            auth_login(request, user)
            request.session['id'] = user.id
            request.session['name'] = user.first_name
            request.session['surname'] = user.last_name

            messages.success(request, f"Welcome to the Criminal Detection System, {user.first_name} {user.last_name}!")
            return redirect('dashboard')  # Make sure 'success' is defined in your urls.py

        else:
            if User.objects.filter(email=email).exists():
                messages.error(request, 'Oops, wrong password. Please try again.')
            else:
                messages.error(request, 'Oops, that personnel ID does not exist.')
            return redirect('/')

    return render(request, 'session/login.html')  # fallback if someone visits this route directly



def dashboard(request):
    # Stats section
    stats = [
        {'label': 'Total Authorized Persons', 'count': AuthorizedPerson.objects.count()},
        {'label': 'Students', 'count': AuthorizedPerson.objects.filter(role='student').count()},
        {'label': 'Staff', 'count': AuthorizedPerson.objects.filter(role='staff').count()},
        {'label': 'Lecturers', 'count': AuthorizedPerson.objects.filter(role='lecturer').count()},
    ]

    # Pie chart data
    role_qs = AuthorizedPerson.objects.values('role').annotate(count=Count('id'))
    role_labels = [x['role'].capitalize() for x in role_qs]
    role_data = [x['count'] for x in role_qs]

    # Detections chart (last 7 days)
    past_week = timezone.now() - timedelta(days=7)
    detections = UnauthorizedDetection.objects.filter(timestamp__gte=past_week)

    detections_by_day = (
        detections
        .annotate(day=TruncDate('timestamp'))  # Gives actual date objects
        .values('day')
        .annotate(count=Count('id'))
        .order_by('day')
    )

    detections_dates = [x['day'].strftime('%b %d') for x in detections_by_day]
    detections_counts = [x['count'] for x in detections_by_day]

    # Trend Analysis Data
    # Hourly trends (last 24 hours)
    past_24h = timezone.now() - timedelta(hours=24)
    hourly_detections = (
        UnauthorizedDetection.objects.filter(timestamp__gte=past_24h)
        .annotate(hour=TruncDate('timestamp'))
        .values('hour')
        .annotate(count=Count('id'))
        .order_by('hour')
    )

    # Access point trends
    access_point_trends = (
        UnauthorizedDetection.objects.all()
        .values('access_attempted')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]  # Top 5 access points
    )

    # Weekly trends (last 4 weeks)
    past_4_weeks = timezone.now() - timedelta(weeks=4)
    weekly_trends_raw = (
        UnauthorizedDetection.objects.filter(timestamp__gte=past_4_weeks)
        .annotate(week=TruncDate('timestamp'))
        .values('week')
        .annotate(count=Count('id'))
        .order_by('week')
    )

    # Format weekly trends for chart
    weekly_trends = []
    for item in weekly_trends_raw:
        weekly_trends.append({
            'week': item['week'].strftime('%b %d'),
            'count': item['count']
        })

    # Detection confidence trends
    confidence_ranges = [
        {'label': 'High (80-100%)', 'min': 0.8, 'max': 1.0, 'color': '#ef4444'},
        {'label': 'Medium (60-79%)', 'min': 0.6, 'max': 0.8, 'color': '#f97316'},
        {'label': 'Low (40-59%)', 'min': 0.4, 'max': 0.6, 'color': '#eab308'},
        {'label': 'Very Low (<40%)', 'min': 0.0, 'max': 0.4, 'color': '#22c55e'},
    ]

    confidence_data = []
    for conf_range in confidence_ranges:
        count = UnauthorizedDetection.objects.filter(
            detection_confidence__gte=conf_range['min'],
            detection_confidence__lt=conf_range['max']
        ).count()
        confidence_data.append({
            'label': conf_range['label'],
            'count': count,
            'color': conf_range['color']
        })

    # Recent trends comparison (this week vs last week)
    this_week_start = timezone.now() - timedelta(days=timezone.now().weekday())
    last_week_start = this_week_start - timedelta(days=7)
    last_week_end = this_week_start

    this_week_count = UnauthorizedDetection.objects.filter(
        timestamp__gte=this_week_start
    ).count()

    last_week_count = UnauthorizedDetection.objects.filter(
        timestamp__gte=last_week_start,
        timestamp__lt=last_week_end
    ).count()

    trend_percentage = 0
    if last_week_count > 0:
        trend_percentage = ((this_week_count - last_week_count) / last_week_count) * 100

    # Context for template
    context = {
        'stats': stats,
        'status_labels': role_labels,
        'status_data': role_data,
        'sightings_dates': detections_dates,
        'sightings_counts': detections_counts,
        'recent_sightings': detections.order_by('-timestamp')[:5],
        'recent_files': File.objects.order_by('-timestamp')[:5],
        # Trend data
        'access_point_trends': list(access_point_trends),
        'weekly_trends': list(weekly_trends),
        'confidence_data': confidence_data,
        'this_week_count': this_week_count,
        'last_week_count': last_week_count,
        'trend_percentage': trend_percentage,
        'total_detections': UnauthorizedDetection.objects.count(),
        'avg_confidence': UnauthorizedDetection.objects.aggregate(avg=Avg('detection_confidence'))['avg'] or 0,
    }

    return render(request, 'home/dashboard.html', context)



def logOut(request):
    logout(request)
    messages.add_message(request,messages.INFO,"Successfully logged out")
    return redirect("login")


def add_user(request):
    return render(request, 'home/add_user.html')

def save_user(request):
    if request.method == 'POST':
        id_number = request.POST.get("id_number")
        name = request.POST.get("name")
        department = request.POST.get("department")
        role = request.POST.get("role", "student")
        profile_picture = request.FILES.get("profile_picture")

        if AuthorizedPerson.objects.filter(id_number=id_number).exists():
            messages.error(request, "Person with that ID already exists.")
            return redirect('add_user')
        if not profile_picture:
            messages.error(request, "Image is required.")
            return redirect('add_user')

        # Save file
        fs = FileSystemStorage()
        filename = fs.save(profile_picture.name, profile_picture)
        uploaded_file_path = fs.path(filename)
        uploaded_file_url = fs.url(filename)

        # Process the image for face recognition
        face_data = process_authorized_face(uploaded_file_path, name)

        if not face_data['face_detected']:
            # If no face detected, still save but mark as not ready for recognition
            messages.warning(request, f"Warning: {face_data['error']} The profile was saved but won't be used for real-time detection until a proper face image is uploaded.")
        elif face_data['face_quality_score'] < 0.5:
            messages.warning(request, f"Warning: The uploaded image has low quality (Score: {face_data['face_quality_score']:.2f}). Consider uploading a clearer image for better detection accuracy.")
        else:
            messages.success(request, f"Authorized person successfully added with high-quality face recognition data (Score: {face_data['face_quality_score']:.2f}).")

        # Create authorized person record with face data
        AuthorizedPerson.objects.create(
            name=name,
            id_number=id_number,
            department=department,
            role=role,
            profile_picture=uploaded_file_url.lstrip('/'),
            face_embedding=face_data['face_embedding'],
            face_embeddings_backup=face_data['face_embeddings_backup'],
            face_quality_score=face_data['face_quality_score'],
            face_detected=face_data['face_detected'],
            face_detection_confidence=face_data['face_detection_confidence'],
            image_processed=True,
            processing_notes=face_data['processing_notes']
        )

        return redirect('view_users')



# view to get authorized persons details
def view_users(request):
    users = AuthorizedPerson.objects.all()
    context = {
        "users": users
    }
    return render(request, 'home/view_users.html', context)



@csrf_exempt
def change_user_status(request):
    if request.method == 'POST':
        user_id = request.POST.get('user_id')
        is_active = request.POST.get('is_active') == 'true'

        try:
            user = AuthorizedPerson.objects.get(id=user_id)
            user.is_active = is_active
            user.save()
            return JsonResponse({'success': True})
        except AuthorizedPerson.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'User not found'})
    return JsonResponse({'success': False, 'message': 'Invalid request'})





def detect_unauthorized(request):
    user = User.objects.get(id=request.session['id'])
    context = {
        "user": user
    }
    return render(request, 'home/detect_unauthorized.html', context)


def spotted_unauthorized(request):
    # Fetch all unauthorized detections
    detections = UnauthorizedDetection.objects.all().order_by('-timestamp')
    return render(request, 'home/spotted_unauthorized.html', {'detections': detections})



def mark_detection_handled(request, detection_id):
    # Mark unauthorized detection as handled
    detection = get_object_or_404(UnauthorizedDetection, pk=detection_id)
    detection.handled = True  # Assuming we add a handled field, or just delete
    detection.save()

    messages.success(request, f"Detection for {detection.detected_name or 'Unknown Person'} has been marked as handled.")
    return redirect("spotted_unauthorized")

def test_detection_system(request):
    """Test endpoint to verify the detection system is working"""
    try:
        # Check if there are any authorized persons
        authorized_count = AuthorizedPerson.objects.filter(is_active=True).count()
        total_detections = UnauthorizedDetection.objects.count()

        # Test SMS configuration
        sms_config = SMSConfiguration.objects.filter(is_active=True).first()
        sms_status = "Configured" if sms_config else "Not configured"

        # Test alert recipients
        recipients_count = AlertRecipient.objects.filter(is_active=True, alert_for_unauthorized=True).count()

        test_results = {
            'authorized_persons': authorized_count,
            'total_detections': total_detections,
            'sms_config': sms_status,
            'alert_recipients': recipients_count,
            'system_status': 'Ready' if authorized_count > 0 else 'No authorized persons registered'
        }

        return JsonResponse(test_results)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def process_frame(request):
    """API endpoint to process frames from browser camera"""
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)

    try:
        if 'frame' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No frame provided'}, status=400)

        frame_file = request.FILES['frame']

        # Save the uploaded frame temporarily
        fs = FileSystemStorage()
        filename = fs.save(f'temp_frame_{uuid.uuid4()}.jpg', frame_file)
        frame_path = fs.path(filename)

        # Process the frame for face detection
        face_data = process_authorized_face(frame_path, 'frame')

        detections = []

        if face_data['face_detected']:
            # Check if this face matches any authorized person
            authorized_persons = AuthorizedPerson.objects.filter(
                is_active=True,
                face_detected=True,
                face_embedding__isnull=False
            )

            best_match = None
            best_confidence = 0.0

            for person in authorized_persons:
                if person.face_embedding:
                    # Calculate similarity
                    distance = np.linalg.norm(
                        np.array(person.face_embedding) - np.array(face_data['face_embedding'])
                    )
                    confidence = 1.0 - min(distance, 1.0)

                    if confidence > best_confidence and confidence > 0.4:  # Threshold
                        best_confidence = confidence
                        best_match = person

            if best_match:
                # Authorized person detected
                detections.append({
                    'id': best_match.id,
                    'name': best_match.name,
                    'type': 'authorized',
                    'role': best_match.role,
                    'confidence': best_confidence,
                    'timestamp': timezone.now().strftime('%H:%M:%S'),
                    'area': 'lecture_hall',
                    'coordinates': '5.6037, -0.1870',
                    'handled': False
                })
            else:
                # Unauthorized person detected
                # Check for duplicates (same face in last 30 seconds)
                recent_detection = UnauthorizedDetection.objects.filter(
                    timestamp__gte=timezone.now() - timedelta(seconds=30),
                    detection_confidence__gt=0.3
                ).first()

                if not recent_detection:
                    # Create new unauthorized detection
                    detection = UnauthorizedDetection.objects.create(
                        detected_name='Unknown Person',
                        latitude=5.6037,
                        longitude=-0.1870,
                        detection_address='University Campus',
                        detection_confidence=face_data['face_detection_confidence'],
                        camera_location='Browser Camera',
                        detection_method='browser_camera',
                        access_attempted='lecture_hall'
                    )

                    # Send SMS alert
                    try:
                        sms_service = ArkeselSMSService()
                        image_url = None
                        sms_result = sms_service.send_unauthorized_alert(detection, image_url)

                        if sms_result['success']:
                            detection.sms_alert_sent = True
                            detection.sms_sent_at = timezone.now()
                            detection.sms_recipients = [r['phone'] for r in sms_result['results'] if r['success']]
                            detection.sms_response = sms_result
                            detection.save()
                    except Exception as sms_error:
                        print(f"SMS error: {sms_error}")

                    detections.append({
                        'id': detection.id,
                        'name': 'Unknown Person',
                        'type': 'unauthorized',
                        'confidence': face_data['face_detection_confidence'],
                        'timestamp': detection.timestamp.strftime('%H:%M:%S'),
                        'area': detection.access_attempted,
                        'coordinates': f"{detection.latitude}, {detection.longitude}",
                        'handled': detection.handled
                    })

        # Clean up temporary file
        try:
            fs.delete(filename)
        except:
            pass

        # Get total detection count
        total_detections = UnauthorizedDetection.objects.filter(
            timestamp__gte=timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ).count()

        return JsonResponse({
            'success': True,
            'detections': detections,
            'total_detections': total_detections,
            'face_detected': face_data['face_detected']
        })

    except Exception as e:
        print(f"Frame processing error: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e),
            'detections': [],
            'total_detections': 0
        }, status=500)

def get_recent_detections(request):
    """API endpoint to get recent unauthorized detections for real-time updates"""
    try:
        # Clean up old detections from memory periodically
        clear_old_detections()

        # Get recent detections from the last 5 minutes from database
        db_detections = UnauthorizedDetection.objects.filter(
            timestamp__gte=timezone.now() - timedelta(minutes=5)
        ).order_by('-timestamp')[:10]  # Last 10 detections

        detections_data = []
        for detection in db_detections:
            detections_data.append({
                'id': detection.id,
                'name': detection.detected_name or 'Unknown Person',
                'timestamp': detection.timestamp.strftime('%H:%M:%S'),
                'confidence': float(detection.detection_confidence) if detection.detection_confidence else 0.0,
                'location': detection.detection_address,
                'coordinates': f"{detection.latitude}, {detection.longitude}",
                'area': detection.access_attempted,
                'handled': detection.handled
            })

        # Add in-memory detections from active camera session
        global recent_detections
        for detection in recent_detections[:10]:  # Take first 10 from memory
            # Avoid duplicates by checking if ID already exists
            if not any(d['id'] == detection['id'] for d in detections_data):
                detections_data.insert(0, detection)  # Add to beginning

        # Keep only the most recent 10
        detections_data = detections_data[:10]

        # Get total count for the session (from database + memory)
        db_count = UnauthorizedDetection.objects.filter(
            timestamp__gte=timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        ).count()
        total_count = db_count + len(recent_detections)

        return JsonResponse({
            'success': True,
            'detections': detections_data,
            'total_count': total_count,
            'recent_count': len(detections_data)
        })

    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'detections': [],
            'total_count': 0,
            'recent_count': 0
        }, status=500)





def detect_image(request):
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_path = fs.path(filename)

        # Load known authorized persons using enhanced embeddings
        known_embeddings = []
        known_names = []
        known_persons = {}
        embedding_weights = []

        # Load high-quality authorized person profiles
        persons = AuthorizedPerson.objects.filter(
            face_detected=True,
            face_embedding__isnull=False,
            face_quality_score__gt=0.5,
            is_active=True
        ).order_by('-face_quality_score')
        for person in persons:
            if person.face_embedding:
                primary_embedding = np.array(person.face_embedding, dtype=np.float32)
                known_embeddings.append(primary_embedding)
                known_names.append(f"{person.name} ({person.id_number})")
                known_persons[len(known_embeddings)-1] = person
                embedding_weights.append(person.face_quality_score or 1.0)

        # Fallback for older profiles without embeddings
        fallback_persons = AuthorizedPerson.objects.exclude(
            face_detected=True,
            face_embedding__isnull=False,
            face_quality_score__gt=0.5
        ).filter(is_active=True)
        for person in fallback_persons:
            try:
                img = cv2.imread(person.profile_picture.path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = face_app.get(img)
                if len(faces) > 0:
                    embedding = faces[0].normed_embedding
                    known_embeddings.append(embedding)
                    known_names.append(f"{person.name} ({person.id_number}) [Legacy]")
                    known_persons[len(known_embeddings)-1] = person
                    embedding_weights.append(0.7)  # Lower weight for fallback
            except Exception as e:
                print(f"Error loading person face: {e}")
                continue

        # Load uploaded image
        unknown_img = cv2.imread(uploaded_file_path)
        if unknown_img is None:
            messages.error(request, "Could not read uploaded image.")
            return redirect('detect_unauthorized')

        unknown_img_rgb = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(unknown_img_rgb)

        # Convert to PIL for drawing
        pil_image = Image.fromarray(unknown_img_rgb)
        draw = ImageDraw.Draw(pil_image)

        authorized_found = False
        unauthorized_found = False
        matches_found = []

        for face in faces:
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            embedding = face.normed_embedding

            name = "Unknown"
            confidence = 0.0
            is_authorized = False

            # Enhanced matching with confidence scoring
            if known_embeddings:
                best_confidence = 0.0
                best_match_idx = -1

                for idx, known_emb in enumerate(known_embeddings):
                    distance = np.linalg.norm(known_emb - embedding)
                    weighted_confidence = (1.0 - min(distance, 1.0)) * embedding_weights[idx]

                    if weighted_confidence > best_confidence and distance < 0.65:  # Adjusted threshold
                        best_confidence = weighted_confidence
                        best_match_idx = idx

                if best_match_idx >= 0 and best_confidence > 0.4:
                    name = known_names[best_match_idx]
                    confidence = best_confidence
                    person = known_persons[best_match_idx]
                    is_authorized = True
                    authorized_found = True
                    matches_found.append({
                        'name': person.name,
                        'confidence': confidence,
                        'role': person.role,
                        'person_obj': person
                    })
                else:
                    unauthorized_found = True

            # Draw bounding box and label
            if is_authorized:
                color = (0, 255, 0)  # Green for authorized
                label = f"Authorized: {name} ({confidence:.2f})"
            else:
                color = (255, 0, 0)  # Red for unauthorized
                label = "Unauthorized Person"

            draw.rectangle(((left, top), (right, bottom)), outline=color, width=3)

            try:
                font = ImageFont.load_default()
            except:
                font = None

            # Calculate text size for better positioning
            try:
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                # Fallback for older PIL versions
                text_width, text_height = draw.textsize(label, font=font)

            # Draw background rectangle for text
            draw.rectangle(((left, top - text_height - 10), (left + text_width + 10, top)), fill=color)
            draw.text((left + 5, top - text_height - 5), label, fill=(255, 255, 255), font=font)

        del draw

        # Save the result image
        result_path = uploaded_file_path.replace('.', '_result.')
        pil_image.save(result_path)

        # Enhanced feedback messages with detection creation and SMS alerts
        if authorized_found:
            messages.success(request, f"✅ AUTHORIZED PERSON(S) DETECTED: {', '.join([m['name'] for m in matches_found])} - Access granted.")
        elif unauthorized_found:
            sms_alerts_sent = 0
            detections_created = 0

            # Get camera location (for uploaded images, we use "Manual Upload")
            location_data = {
                'description': 'Manual Photo Upload',
                'latitude': 5.6037,  # Default Accra coordinates
                'longitude': -0.1870,
                'address': 'Photo Upload Detection System'
            }

            # Save uploaded image as evidence
            captured_image = None
            try:
                # Copy the uploaded file to detections directory
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                evidence_filename = f"upload_detection_unauthorized_{timestamp}.jpg"

                # Read the uploaded image and save as evidence
                with open(uploaded_file_path, 'rb') as f:
                    image_content = f.read()
                    captured_image = ContentFile(image_content, name=evidence_filename)

            except Exception as img_error:
                print(f"Error saving evidence image: {img_error}")

            # Create detection record
            detection = UnauthorizedDetection.objects.create(
                latitude=location_data['latitude'],
                longitude=location_data['longitude'],
                detection_address=location_data['address'],
                camera_location=location_data['description'],
                detection_method='uploaded_image'
            )

            # Save evidence image to the detection
            if captured_image:
                detection.image_captured = captured_image
                detection.save()

            detections_created += 1
            print(f"Created unauthorized detection record")

            # Send SMS Alert via Arkesel
            try:
                sms_service = ArkeselSMSService()

                # Create image URL if evidence image exists
                image_url = None
                if detection.image_captured:
                    image_url = f"{settings.MEDIA_URL}{detection.image_captured.name}"
                    if not image_url.startswith('http'):
                        # Add domain for absolute URL (adjust as needed)
                        image_url = f"http://localhost:8000{image_url}"

                # Send SMS alert
                sms_result = sms_service.send_unauthorized_alert(detection, image_url)

                if sms_result['success']:
                    # Update detection with SMS info
                    detection.sms_alert_sent = True
                    detection.sms_sent_at = timezone.now()
                    detection.sms_recipients = [r['phone'] for r in sms_result['results'] if r['success']]
                    detection.sms_response = sms_result
                    detection.save()

                    sms_alerts_sent += sms_result['total_sent']
                    print(f"SMS Alert sent successfully: {sms_result['total_sent']} recipients")

                else:
                    print(f"SMS Alert failed: {sms_result.get('error', 'Unknown error')}")
                    detection.sms_response = sms_result
                    detection.save()

            except Exception as sms_error:
                print(f"SMS service error: {sms_error}")
                detection.sms_response = {'error': str(sms_error)}
                detection.save()

            # Success message for unauthorized detection
            base_message = f"🚨 UNAUTHORIZED ACCESS DETECTED"

            if detections_created > 0:
                base_message += f" | ✅ Detection recorded"
            if sms_alerts_sent > 0:
                base_message += f" | 📱 SMS alerts sent to {sms_alerts_sent} recipients"
            base_message += " | 🚔 Security team notified!"

            messages.warning(request, base_message)
        else:
            messages.info(request, "No faces detected in the uploaded image or image quality is insufficient.")

    return redirect('detect_unauthorized')
# Global detection storage (for browser-based camera)
recent_detections = []  # Global list to store recent detections

class VideoCamera:
    def __init__(self):
        print("📹 Initializing VideoCamera...")
        self.video = None
        self.initialized = False

        try:
            # Try different camera indices if 0 doesn't work
            for camera_index in [0, 1, 2]:
                print(f"📹 Trying camera index {camera_index}...")
                self.video = cv2.VideoCapture(camera_index)
                if self.video.isOpened():
                    print(f"✅ Camera index {camera_index} opened successfully")
                    break
                else:
                    self.video.release()
                    self.video = None

            if self.video is None or not self.video.isOpened():
                raise Exception("No camera device found or accessible")

            # Set camera properties
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.video.set(cv2.CAP_PROP_FPS, 30)
            self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time

            # Verify camera is working
            ret, test_frame = self.video.read()
            if not ret or test_frame is None:
                raise Exception("Camera opened but cannot capture frames")

            print("✅ Camera initialized and tested successfully")
            self.initialized = True

        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            if self.video:
                self.video.release()
            self.video = None
            self.initialized = False
            raise e

        # Load known authorized persons once using enhanced embeddings
        self.known_embeddings = []
        self.known_backup_embeddings = []
        self.known_names = []
        self.known_national_ids = []
        self.known_criminals = {}
        self.embedding_weights = []  # Quality-based weights for embeddings
        self.load_authorized_persons_enhanced()
        
    def load_authorized_persons_enhanced(self):
        """Load all authorized person data using pre-processed embeddings for maximum accuracy"""
        persons = AuthorizedPerson.objects.filter(
            face_detected=True,
            face_embedding__isnull=False,
            face_quality_score__gt=0.5,
            is_active=True
        ).order_by('-face_quality_score')
        loaded_count = 0
        fallback_count = 0

        for person in persons:
            try:
                # Use pre-processed embeddings if available
                if person.face_embedding and person.face_detected:
                    # Add primary embedding
                    primary_embedding = np.array(person.face_embedding, dtype=np.float32)
                    self.known_embeddings.append(primary_embedding)
                    self.known_names.append(person.name)
                    self.known_national_ids.append(person.id_number)
                    self.known_criminals[person.id_number] = person
                    self.embedding_weights.append(person.face_quality_score or 1.0)

                    # Add backup embeddings for better matching
                    backup_embeddings = []
                    if person.face_embeddings_backup:
                        for backup in person.face_embeddings_backup:
                            if isinstance(backup, dict) and 'embedding' in backup:
                                backup_embedding = np.array(backup['embedding'], dtype=np.float32)
                                backup_embeddings.append({
                                    'embedding': backup_embedding,
                                    'enhancement': backup.get('enhancement', 'unknown'),
                                    'confidence': backup.get('confidence', 0.5)
                                })

                    self.known_backup_embeddings.append(backup_embeddings)
                    loaded_count += 1
                    print(f"Loaded enhanced profile for {person.name} (Quality: {person.face_quality_score:.3f}, Backups: {len(backup_embeddings)})")

                else:
                    # Fallback: process image in real-time (less accurate but still works)
                    try:
                        img = cv2.imread(person.profile_picture.path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            faces = face_app.get(img)
                            if len(faces) > 0:
                                self.known_embeddings.append(faces[0].normed_embedding)
                                self.known_names.append(person.name)
                                self.known_national_ids.append(person.id_number)
                                self.known_criminals[person.id_number] = person
                                self.known_backup_embeddings.append([])  # No backups
                                self.embedding_weights.append(0.7)  # Lower weight for fallback
                                fallback_count += 1
                                print(f"Loaded fallback profile for {person.name}")
                    except Exception as e:
                        print(f"Fallback loading failed for {person.name}: {e}")

            except Exception as e:
                print(f"Error loading person {person.name}: {e}")

        print(f"🚀 Authorized person recognition system loaded: {loaded_count} enhanced profiles, {fallback_count} fallback profiles")
        print(f"📊 Total known persons: {len(self.known_embeddings)}")
        print(f"🎯 Recognition ready for {len([p for p in persons if p.is_active])} active authorized persons")

        # Load non-recognition-ready persons for reference
        non_ready_persons = AuthorizedPerson.objects.exclude(
            face_detected=True,
            face_embedding__isnull=False,
            face_quality_score__gt=0.5
        ).filter(is_active=True).count()
        if non_ready_persons > 0:
            print(f"Warning: {non_ready_persons} authorized person profiles are not ready for recognition (poor image quality or no face detected)")

    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()

    def get_frame(self):
        # Check if camera is properly initialized
        if not self.initialized or self.video is None or not self.video.isOpened():
            print("❌ Camera not properly initialized")
            return None, []

        try:
            success, image = self.video.read()
            if not success or image is None:
                print("❌ Failed to read frame from camera")
                return None, []

            # Process the frame for face detection
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = face_app.get(rgb_frame)

            if len(faces) > 0:
                print(f"📷 Detected {len(faces)} face(s) in frame")
            else:
                print("📷 No faces detected in frame")
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return None, []
        
        detection_results = []
        
        for face in faces:
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            embedding = face.normed_embedding
            
            name = "Unknown"
            status = "Unknown"
            confidence = 0.0
            best_match_info = None
            is_authorized = False

            # Check for authorized person match
            if self.known_embeddings:
                print(f"🔍 Matching against {len(self.known_embeddings)} known embeddings")
                # Enhanced matching using multiple embeddings and weighted scoring
                best_match_info = self.enhanced_face_matching(embedding)

                if best_match_info and best_match_info['confidence'] > 0.4:  # Adjusted threshold
                    id_number = best_match_info['national_id']
                    person_obj = self.known_criminals[id_number]
                    name = person_obj.name
                    status = "Authorized"
                    confidence = best_match_info['confidence']
                    is_authorized = True
                    print(f"✅ AUTHORIZED: {name} (ID: {id_number}, Confidence: {confidence:.3f})")
                else:
                    # No authorized match found - treat as unauthorized
                    status = "Unauthorized"
                    confidence = best_match_info['confidence'] if best_match_info else 0.0
                    is_authorized = False
                    name = "Unknown Person"
                    print(f"🚨 UNAUTHORIZED DETECTED: Confidence: {confidence:.3f}")
            else:
                # No authorized persons loaded - all detections are unauthorized
                print("⚠️ No authorized persons loaded - all detections will be marked as unauthorized")
                best_match_info = None
                status = "Unauthorized"
                confidence = 0.0
                is_authorized = False
                name = "Unknown Person"
                print(f"🚨 UNAUTHORIZED DETECTED: No authorized database loaded")

            # Handle unauthorized detection
            if not is_authorized:
                # Check for duplicates before creating new detection
                duplicate_detection = None
                if embedding is not None:
                    # Create a temporary detection object to check for duplicates
                    temp_detection = UnauthorizedDetection(
                        face_embedding=embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                    )
                    duplicate_detection = temp_detection.check_duplicate(embedding)

                if duplicate_detection:
                    print(f"🔄 DUPLICATE: Skipping detection - similar to #{duplicate_detection.id} from {duplicate_detection.timestamp}")
                    status = "Unauthorized (Duplicate)"
                else:
                    # Get camera location with dynamic area detection
                    location_data = get_camera_location()

                    # Save captured frame as evidence
                    captured_image = save_captured_image(image, "Unauthorized_Person",
                                                       f"unauthorized_{int(timezone.now().timestamp())}")

                    # Create detection record
                    detection = UnauthorizedDetection.objects.create(
                        latitude=location_data['latitude'],
                        longitude=location_data['longitude'],
                        detection_address=location_data['address'],
                        detection_confidence=confidence,
                        camera_location=location_data['description'],
                        detection_method='realtime_webcam',
                        access_attempted=location_data.get('area_name', 'lecture_hall'),
                        face_embedding=embedding.tolist() if embedding is not None and hasattr(embedding, 'tolist') else embedding
                    )

                    # Save captured image to the detection
                    if captured_image:
                        detection.image_captured = captured_image
                        detection.save()

                    print(f"🚨 Recorded unauthorized detection (Confidence: {confidence:.3f}) - Area: {detection.access_attempted}")

                    # Add to recent detections for real-time display
                    global recent_detections
                    detection_data = {
                        'id': detection.id,
                        'name': detection.detected_name or 'Unknown Person',
                        'timestamp': detection.timestamp.strftime('%H:%M:%S'),
                        'confidence': float(detection.detection_confidence) if detection.detection_confidence else 0.0,
                        'location': detection.detection_address,
                        'coordinates': f"{detection.latitude}, {detection.longitude}",
                        'area': detection.access_attempted,
                        'handled': detection.handled
                    }
                    recent_detections.insert(0, detection_data)  # Add to beginning

                    # Keep only last 50 detections in memory
                    if len(recent_detections) > 50:
                        recent_detections = recent_detections[:50]

                    # Send SMS Alert via Arkesel
                    try:
                        sms_service = ArkeselSMSService()

                        # Create image URL if captured image exists
                        image_url = None
                        if detection.image_captured:
                            image_url = f"{settings.MEDIA_URL}{detection.image_captured.name}"
                            if not image_url.startswith('http'):
                                # Add domain for absolute URL (adjust as needed)
                                image_url = f"http://localhost:8000{image_url}"

                        # Send SMS alert
                        sms_result = sms_service.send_unauthorized_alert(detection, image_url)

                        if sms_result['success']:
                            # Update detection with SMS info
                            detection.sms_alert_sent = True
                            detection.sms_sent_at = timezone.now()
                            detection.sms_recipients = [r['phone'] for r in sms_result['results'] if r['success']]
                            detection.sms_response = sms_result
                            detection.save()

                            print(f"📱 SMS Alert sent successfully: {sms_result['total_sent']} recipients")

                            # Store success info for UI feedback
                            best_match_info = best_match_info or {}
                            best_match_info['sms_sent'] = True
                            best_match_info['sms_count'] = sms_result['total_sent']
                        else:
                            print(f"❌ SMS Alert failed: {sms_result.get('error', 'Unknown error')}")
                            detection.sms_response = sms_result
                            detection.save()

                            best_match_info = best_match_info or {}
                            best_match_info['sms_sent'] = False
                            best_match_info['sms_error'] = sms_result.get('error', 'SMS failed')

                    except Exception as sms_error:
                        print(f"❌ SMS service error: {sms_error}")
                        detection.sms_response = {'error': str(sms_error)}
                        detection.save()

                        best_match_info = best_match_info or {}
                        best_match_info['sms_sent'] = False
                        best_match_info['sms_error'] = str(sms_error)
            
            # Draw bounding box and label
            if is_authorized:
                color = (0, 255, 0)  # Green for authorized
                label = f"Authorized: {name}"
                if confidence > 0:
                    label += f" ({confidence:.2f})"
                if best_match_info and 'match_type' in best_match_info:
                    label += f" [{best_match_info['match_type']}]"
            else:
                color = (0, 0, 255)  # Red for unauthorized
                label = "Unauthorized Person"
                if best_match_info and 'sms_sent' in best_match_info and best_match_info['sms_sent']:
                    label += " (Alert Sent)"

            cv2.rectangle(image, (left, top), (right, bottom), color, 2)

            # Draw label background with adaptive size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (left, top - 20), (left + label_size[0] + 5, top), color, -1)
            cv2.putText(image, label, (left + 2, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            detection_results.append({
                'name': name,
                'status': status,
                'confidence': confidence,
                'bbox': [left, top, right, bottom],
                'match_info': best_match_info
            })
        
        # Convert image to JPEG
        ret, jpeg = cv2.imencode('.jpg', image)
        if ret:
            return jpeg.tobytes(), detection_results
        return None, []
    
    def enhanced_face_matching(self, query_embedding):
        """
        Enhanced face matching using multiple embeddings and weighted scoring
        """
        best_confidence = 0.0
        best_match = None

        for idx, primary_embedding in enumerate(self.known_embeddings):
            id_number = self.known_national_ids[idx]
            embedding_weight = self.embedding_weights[idx]
            backup_embeddings = self.known_backup_embeddings[idx]

            # Calculate distance to primary embedding
            primary_distance = np.linalg.norm(primary_embedding - query_embedding)
            primary_confidence = (1.0 - min(primary_distance, 1.0)) * embedding_weight

            max_confidence = primary_confidence
            match_type = "primary"

            # Check backup embeddings for better matches
            for backup in backup_embeddings:
                backup_embedding = backup['embedding']
                backup_distance = np.linalg.norm(backup_embedding - query_embedding)
                backup_confidence = (1.0 - min(backup_distance, 1.0)) * embedding_weight * 0.9  # Slight penalty for backup

                if backup_confidence > max_confidence:
                    max_confidence = backup_confidence
                    match_type = f"backup-{backup['enhancement']}"

            # Apply additional scoring factors
            person_obj = self.known_criminals[id_number]

            # Bonus for high-quality profiles
            if hasattr(person_obj, 'face_quality_score') and person_obj.face_quality_score:
                quality_bonus = person_obj.face_quality_score * 0.1
                max_confidence += quality_bonus

            # Active status bonus
            if hasattr(person_obj, 'is_active') and person_obj.is_active:
                max_confidence *= 1.05  # Slight bonus for active authorized persons

            if max_confidence > best_confidence:
                best_confidence = max_confidence
                best_match = {
                    'national_id': id_number,
                    'confidence': best_confidence,
                    'match_type': match_type,
                    'primary_distance': primary_distance,
                    'embedding_weight': embedding_weight
                }

        return best_match if best_confidence > 0.4 else None


# SMS Configuration Management Views

def sms_configuration(request):
    """View for SMS configuration settings"""
    config = SMSConfiguration.objects.filter(is_active=True).first()
    context = {
        'config': config,
        'user': User.objects.get(id=request.session['id']) if request.session.get('id') else None
    }
    return render(request, 'home/sms_configuration.html', context)

def save_sms_configuration(request):
    """Save SMS configuration settings"""
    if request.method == 'POST':
        api_key = request.POST.get('api_key')
        sender_id = request.POST.get('sender_id')
        send_alerts_enabled = request.POST.get('send_alerts_enabled') == 'on'
        alert_delay_minutes = request.POST.get('alert_delay_minutes', 5)
        include_location = request.POST.get('include_location') == 'on'
        include_image_link = request.POST.get('include_image_link') == 'on'
        
        if not api_key or not sender_id:
            messages.error(request, "API Key and Sender ID are required.")
            return redirect('sms_configuration')
        
        # Deactivate existing configs
        SMSConfiguration.objects.update(is_active=False)
        
        # Create new config
        SMSConfiguration.objects.create(
            api_key=api_key,
            sender_id=sender_id,
            send_alerts_enabled=send_alerts_enabled,
            alert_delay_minutes=int(alert_delay_minutes),
            include_location=include_location,
            include_image_link=include_image_link,
            is_active=True
        )
        
        messages.success(request, "SMS configuration saved successfully.")
        
    return redirect('sms_configuration')

def test_sms_service(request):
    """Test SMS service functionality"""
    if request.method == 'POST':
        test_phone = request.POST.get('test_phone')
        
        if not test_phone:
            messages.error(request, "Please provide a phone number for testing.")
            return redirect('sms_configuration')
        
        try:
            sms_service = ArkeselSMSService()
            
            # Check balance first
            balance_result = sms_service.check_balance()
            if not balance_result['success']:
                messages.error(request, f"SMS service connection failed: {balance_result['error']}")
                return redirect('sms_configuration')
            
            # Send test message
            test_message = f"🔔 SMS Alert System Test\n\nThis is a test message from the Criminal Detection System.\n\nTime: {timezone.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nIf you receive this, SMS alerts are working correctly."
            
            result = sms_service.send_sms(test_phone, test_message)
            
            if result['success']:
                messages.success(request, f"Test SMS sent successfully to {test_phone}. Current balance: {balance_result.get('balance', 'Unknown')}")
            else:
                messages.error(request, f"Test SMS failed: {result['error']}")
                
        except Exception as e:
            messages.error(request, f"SMS test failed: {str(e)}")
    
    return redirect('sms_configuration')

def alert_recipients(request):
    """View for managing alert recipients"""
    recipients = AlertRecipient.objects.all().order_by('priority_level', 'name')
    context = {
        'recipients': recipients,
        'user': User.objects.get(id=request.session['id']) if request.session.get('id') else None
    }
    return render(request, 'home/alert_recipients.html', context)

def save_alert_recipient(request):
    """Save or update alert recipient"""
    if request.method == 'POST':
        recipient_id = request.POST.get('recipient_id')
        name = request.POST.get('name')
        phone_number = request.POST.get('phone_number')
        department = request.POST.get('department', '')
        priority_level = request.POST.get('priority_level', 'medium')
        is_active = request.POST.get('is_active') == 'on'
        
        # Alert preferences
        alert_for_wanted = request.POST.get('alert_for_wanted') == 'on'
        alert_for_investigation = request.POST.get('alert_for_investigation') == 'on'
        alert_for_escaped = request.POST.get('alert_for_escaped') == 'on'
        
        if not name or not phone_number:
            messages.error(request, "Name and phone number are required.")
            return redirect('alert_recipients')
        
        # Validate phone number format
        if not phone_number.startswith('+'):
            messages.error(request, "Phone number must be in international format (e.g., +233123456789).")
            return redirect('alert_recipients')
        
        try:
            if recipient_id:
                # Update existing recipient
                recipient = AlertRecipient.objects.get(id=recipient_id)
                recipient.name = name
                recipient.phone_number = phone_number
                recipient.department = department
                recipient.priority_level = priority_level
                recipient.is_active = is_active
                recipient.alert_for_wanted = alert_for_wanted
                recipient.alert_for_investigation = alert_for_investigation
                recipient.alert_for_escaped = alert_for_escaped
                recipient.save()
                messages.success(request, f"Recipient {name} updated successfully.")
            else:
                # Create new recipient
                AlertRecipient.objects.create(
                    name=name,
                    phone_number=phone_number,
                    department=department,
                    priority_level=priority_level,
                    is_active=is_active,
                    alert_for_wanted=alert_for_wanted,
                    alert_for_investigation=alert_for_investigation,
                    alert_for_escaped=alert_for_escaped
                )
                messages.success(request, f"Recipient {name} added successfully.")
                
        except AlertRecipient.DoesNotExist:
            messages.error(request, "Recipient not found.")
        except Exception as e:
            messages.error(request, f"Error saving recipient: {str(e)}")
    
    return redirect('alert_recipients')

def delete_alert_recipient(request, recipient_id):
    """Delete alert recipient"""
    try:
        recipient = AlertRecipient.objects.get(id=recipient_id)
        name = recipient.name
        recipient.delete()
        messages.success(request, f"Recipient {name} deleted successfully.")
    except AlertRecipient.DoesNotExist:
        messages.error(request, "Recipient not found.")
    except Exception as e:
        messages.error(request, f"Error deleting recipient: {str(e)}")
    
    return redirect('alert_recipients')



def webcam_stream(request):
    """Render the webcam streaming page - camera access handled by browser"""
    context = {
        'user': User.objects.get(id=request.session['id']) if request.session.get('id') else None
    }
    return render(request, 'home/webcam_stream.html', context)

def clear_old_detections():
    """Clear old detections from memory to prevent memory leaks"""
    global recent_detections
    # Keep only detections from the last 30 minutes
    cutoff_time = timezone.now() - timedelta(minutes=30)
    recent_detections = [
        detection for detection in recent_detections
        if 'timestamp' in detection and detection['timestamp'] > cutoff_time.strftime('%H:%M:%S')
    ]

def stop_webcam(request):
    """Stop the webcam session - camera cleanup handled by browser"""
    global recent_detections

    print("🛑 Stopping webcam session...")

    # Clear recent detections to free memory
    recent_detections.clear()
    print("🧹 Recent detections cleared")

    messages.success(request, "Webcam session ended successfully.")
    return redirect('detect_unauthorized')







# View to detect criminals using webcam (Updated to use streaming)
def detect_with_webcam(request):
    """Redirect to the new streaming webcam interface"""
    return redirect('webcam_stream')




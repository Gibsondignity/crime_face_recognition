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
from .models import User, Criminal, CriminalLastSpotted, File
from django.contrib.auth import authenticate, login as auth_login
from django.db.models import Count
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
    
    def send_criminal_alert(self, criminal, sighting, captured_image_url=None):
        """Send SMS alert for criminal detection"""
        if not self.config or not self.config.send_alerts_enabled:
            return {'success': False, 'error': 'SMS alerts disabled or not configured'}
        
        # Get recipients based on criminal status
        recipients = self.get_alert_recipients(criminal.status)
        if not recipients:
            return {'success': False, 'error': 'No active recipients found'}
        
        # Create alert message
        message = self.create_alert_message(criminal, sighting, captured_image_url)
        
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
    
    def get_alert_recipients(self, criminal_status):
        """Get recipients based on criminal status"""
        recipients = AlertRecipient.objects.filter(is_active=True)
        
        # Filter based on criminal status preferences
        if criminal_status == 'wanted':
            recipients = recipients.filter(alert_for_wanted=True)
        elif criminal_status == 'under_investigation':
            recipients = recipients.filter(alert_for_investigation=True)
        elif criminal_status == 'escaped':
            recipients = recipients.filter(alert_for_escaped=True)
        
        return recipients.order_by('priority_level')
    
    def create_alert_message(self, criminal, sighting, captured_image_url=None):
        """Create SMS alert message"""
        timestamp = sighting.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"🚨 CRIMINAL ALERT 🚨\n"
        message += f"Name: {criminal.full_name}\n"
        message += f"ID: {criminal.national_id}\n"
        message += f"Status: {criminal.status.upper()}\n"
        message += f"Time: {timestamp}\n"
        
        if self.config.include_location:
            if sighting.camera_location:
                message += f"Location: {sighting.camera_location}\n"
            if sighting.latitude and sighting.longitude:
                message += f"Coordinates: {sighting.latitude}, {sighting.longitude}\n"
                if hasattr(sighting, 'google_maps_link'):
                    message += f"Map: {sighting.google_maps_link}\n"
        
        if captured_image_url and self.config.include_image_link:
            message += f"Image: {captured_image_url}\n"
        
        message += f"Confidence: {sighting.detection_confidence:.2f}\n" if sighting.detection_confidence else ""
        message += "Take immediate action. Contact dispatch for details."
        
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

def save_captured_image(frame, criminal_name, sighting_id):
    """Save captured frame as evidence image"""
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"detection_{criminal_name.replace(' ', '_')}_{timestamp}_{sighting_id}.jpg"
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ret:
            image_content = ContentFile(buffer.tobytes(), name=filename)
            return image_content
    except Exception as e:
        print(f"Error saving captured image: {e}")
    
    return None

def get_camera_location():
    """Get camera location description (can be enhanced with GPS)"""
    # This can be enhanced to get actual GPS coordinates
    # For now, return a default location
    return {
        'description': 'Security Camera - Main Entrance',
        'latitude': 5.6037,  # Accra coordinates
        'longitude': -0.1870,
        'address': 'University Campus, Accra, Ghana'
    }

from datetime import datetime
import json
import base64
from skimage import exposure, filters
from PIL import ImageEnhance

def process_criminal_face(image_path, full_name):
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
        {'label': 'Total Criminals', 'count': Criminal.objects.count()},
        {'label': 'Wanted', 'count': Criminal.objects.filter(status='wanted').count()},
        {'label': 'Arrested', 'count': Criminal.objects.filter(status='arrested').count()},
        {'label': 'Under Investigation', 'count': Criminal.objects.filter(status='under_investigation').count()},
    ]

    # Pie chart data
    status_qs = Criminal.objects.values('status').annotate(count=Count('id'))
    status_labels = [x['status'].capitalize().replace('_', ' ') for x in status_qs]
    status_data = [x['count'] for x in status_qs]

    # Sightings chart (last 7 days)
    past_week = timezone.now() - timedelta(days=7)
    sightings = CriminalLastSpotted.objects.filter(timestamp__gte=past_week)

    sightings_by_day = (
        sightings
        .annotate(day=TruncDate('timestamp'))  # Gives actual date objects
        .values('day')
        .annotate(count=Count('id'))
        .order_by('day')
    )

    sightings_dates = [x['day'].strftime('%b %d') for x in sightings_by_day]
    sightings_counts = [x['count'] for x in sightings_by_day]

    # Context for template
    context = {
        'stats': stats,
        'status_labels': status_labels,
        'status_data': status_data,
        'sightings_dates': sightings_dates,
        'sightings_counts': sightings_counts,
        'recent_sightings': sightings.order_by('-timestamp')[:5],
        'recent_files': File.objects.order_by('-timestamp')[:5],
    }

    return render(request, 'home/dashboard.html', context)



def logOut(request):
    logout(request)
    messages.add_message(request,messages.INFO,"Successfully logged out")
    return redirect("login")


def add_criminal(request):
    return render(request, 'home/add_criminal.html')

def save_criminal(request):
    if request.method == 'POST':
        national_id = request.POST.get("national_id")
        full_name = request.POST.get("full_name")
        address = request.POST.get("address")
        profile_picture = request.FILES.get("profile_picture")

        if Criminal.objects.filter(national_id=national_id).exists():
            messages.error(request, "Criminal with that National ID already exists.")
            return redirect('add_criminal')  # Fixed redirect
        if not profile_picture:
            messages.error(request, "Image is required.")
            return redirect('add_criminal')  # Fixed redirect

        # Save file
        fs = FileSystemStorage()
        filename = fs.save(profile_picture.name, profile_picture)
        uploaded_file_path = fs.path(filename)
        uploaded_file_url = fs.url(filename)

        # Process the image for face recognition
        face_data = process_criminal_face(uploaded_file_path, full_name)
        
        if not face_data['face_detected']:
            # If no face detected, still save but mark as not ready for recognition
            messages.warning(request, f"Warning: {face_data['error']} The profile was saved but won't be used for real-time detection until a proper face image is uploaded.")
        elif face_data['face_quality_score'] < 0.5:
            messages.warning(request, f"Warning: The uploaded image has low quality (Score: {face_data['face_quality_score']:.2f}). Consider uploading a clearer image for better detection accuracy.")
        else:
            messages.success(request, f"Criminal successfully added with high-quality face recognition data (Score: {face_data['face_quality_score']:.2f}).")

        # Create criminal record with face data
        Criminal.objects.create(
            full_name=full_name,
            national_id=national_id,
            address=address,
            profile_picture=uploaded_file_url.lstrip('/'),
            status="wanted",
            face_embedding=face_data['face_embedding'],
            face_embeddings_backup=face_data['face_embeddings_backup'],
            face_quality_score=face_data['face_quality_score'],
            face_detected=face_data['face_detected'],
            face_detection_confidence=face_data['face_detection_confidence'],
            image_processed=True,
            processing_notes=face_data['processing_notes']
        )

        return redirect('view_criminals')



# view to get citizen(criminal) details
def view_criminals(request):
    criminals=Criminal.objects.all()
    context={
        "criminals":criminals
    }
    return render(request,'home/view_criminals.html',context)



@csrf_exempt
def change_criminal_status(request):
    if request.method == 'POST':
        citizen_id = request.POST.get('citizen_id')
        new_status = request.POST.get('new_status')

        try:
            criminal = Criminal.objects.get(id=citizen_id)
            criminal.status = new_status
            criminal.save()
            return JsonResponse({'success': True})
        except Criminal.DoesNotExist:
            return JsonResponse({'success': False, 'message': 'Citizen not found'})
    return JsonResponse({'success': False, 'message': 'Invalid request'})





def identify_criminal(request):
    user = User.objects.get(id=request.session['id'])
    context = {
        "user": user
    }
    return render(request, 'home/identify_criminal.html', context)


def spotted_criminal(request):
    # Fetch all sightings of criminals still marked as 'wanted'
    criminals = CriminalLastSpotted.objects.filter(criminal__status='wanted').order_by('-timestamp')
    return render(request, 'home/spotted_criminal.html', {'criminals': criminals})



def found_criminal(request, thief_id):
    # Update status of all sightings + main criminal record
    sighting = get_object_or_404(CriminalLastSpotted, pk=thief_id)
    criminal = sighting.criminal

    # Update Criminal status
    criminal.status = 'arrested'
    criminal.save()

    # Update all related sightings
    CriminalLastSpotted.objects.filter(criminal=criminal).update(criminal=criminal)

    messages.success(request, f"{criminal.full_name} has been marked as 'arrested'. Good job!")
    return redirect("spotted_criminal")




def detect_image(request):
    if request.method == 'POST' and request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_path = fs.path(filename)

        # Load known criminals using enhanced embeddings
        known_embeddings = []
        known_names = []
        known_criminals = {}
        embedding_weights = []

        # Load high-quality criminal profiles first
        criminals = Criminal.objects.filter(
            face_detected=True,
            face_embedding__isnull=False,
            face_quality_score__gt=0.5
        ).order_by('-face_quality_score')
        for criminal in criminals:
            if criminal.face_embedding:
                primary_embedding = np.array(criminal.face_embedding, dtype=np.float32)
                known_embeddings.append(primary_embedding)
                known_names.append(f"{criminal.full_name} ({criminal.address})")
                known_criminals[len(known_embeddings)-1] = criminal
                embedding_weights.append(criminal.face_quality_score or 1.0)

        # Fallback for older profiles without embeddings
        fallback_criminals = Criminal.objects.exclude(
            face_detected=True,
            face_embedding__isnull=False,
            face_quality_score__gt=0.5
        )
        for criminal in fallback_criminals:
            try:
                img = cv2.imread(criminal.profile_picture.path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = face_app.get(img)
                if len(faces) > 0:
                    embedding = faces[0].normed_embedding
                    known_embeddings.append(embedding)
                    known_names.append(f"{criminal.full_name} ({criminal.address}) [Legacy]")
                    known_criminals[len(known_embeddings)-1] = criminal
                    embedding_weights.append(0.7)  # Lower weight for fallback
            except Exception as e:
                print(f"Error loading criminal face: {e}")
                continue

        # Load uploaded image
        unknown_img = cv2.imread(uploaded_file_path)
        if unknown_img is None:
            messages.error(request, "Could not read uploaded image.")
            return redirect('identify_criminal')

        unknown_img_rgb = cv2.cvtColor(unknown_img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(unknown_img_rgb)

        # Convert to PIL for drawing
        pil_image = Image.fromarray(unknown_img_rgb)
        draw = ImageDraw.Draw(pil_image)

        found = False
        matches_found = []

        for face in faces:
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            embedding = face.normed_embedding

            name = "Unknown"
            confidence = 0.0

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
                    criminal = known_criminals[best_match_idx]
                    found = True
                    matches_found.append({
                        'name': criminal.full_name,
                        'confidence': confidence,
                        'status': criminal.status,
                        'criminal_obj': criminal
                    })

            # Draw bounding box and label
            color = (0, 255, 0) if confidence > 0.8 else (255, 165, 0) if confidence > 0.6 else (255, 0, 0)
            draw.rectangle(((left, top), (right, bottom)), outline=color, width=3)
            
            # Create enhanced label
            if confidence > 0:
                label = f"{name} ({confidence:.2f})"
            else:
                label = "Unknown Person"
                
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

        # Enhanced feedback messages with sighting creation and SMS alerts
        if found:
            sms_alerts_sent = 0
            sightings_created = 0
            
            for match in matches_found:
                criminal = match['criminal_obj']
                confidence = match['confidence']
                
                # Only process if criminal is not released
                if criminal.status != 'released':
                    try:
                        # Check if this criminal was spotted recently (within last 5 minutes)
                        recent_spotting = CriminalLastSpotted.objects.filter(
                            criminal=criminal,
                            timestamp__gte=timezone.now() - timedelta(minutes=5)
                        ).exists()
                        
                        if not recent_spotting:
                            # Get camera location (for uploaded images, we use "Manual Upload")
                            location_data = {
                                'description': 'Manual Photo Upload',
                                'latitude': 5.6037,  # Default Accra coordinates
                                'longitude': -0.1870,
                                'address': 'Photo Upload Detection System'
                            }
                            
                            # Save uploaded image as evidence (copy to sightings folder)
                            captured_image = None
                            try:
                                # Copy the uploaded file to sightings directory
                                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                                evidence_filename = f"upload_detection_{criminal.full_name.replace(' ', '_')}_{timestamp}.jpg"
                                
                                # Read the uploaded image and save as evidence
                                with open(uploaded_file_path, 'rb') as f:
                                    image_content = f.read()
                                    captured_image = ContentFile(image_content, name=evidence_filename)
                                
                            except Exception as img_error:
                                print(f"Error saving evidence image: {img_error}")
                            
                            # Create sighting record
                            sighting = CriminalLastSpotted.objects.create(
                                criminal=criminal,
                                latitude=location_data['latitude'],
                                longitude=location_data['longitude'],
                                spotted_address=location_data['address'],
                                detection_confidence=confidence,
                                camera_location=location_data['description'],
                                detection_method='uploaded_image'
                            )
                            
                            # Save evidence image to the sighting
                            if captured_image:
                                sighting.image_captured = captured_image
                                sighting.save()
                            
                            sightings_created += 1
                            print(f"Created sighting record: {criminal.full_name} (Confidence: {confidence:.3f})")
                            
                            # Send SMS Alert via Arkesel
                            try:
                                sms_service = ArkeselSMSService()
                                
                                # Create image URL if evidence image exists
                                image_url = None
                                if sighting.image_captured:
                                    image_url = f"{settings.MEDIA_URL}{sighting.image_captured.name}"
                                    if not image_url.startswith('http'):
                                        # Add domain for absolute URL (adjust as needed)
                                        image_url = f"http://localhost:8000{image_url}"
                                
                                # Send SMS alert
                                sms_result = sms_service.send_criminal_alert(criminal, sighting, image_url)
                                
                                if sms_result['success']:
                                    # Update sighting with SMS info
                                    sighting.sms_alert_sent = True
                                    sighting.sms_sent_at = timezone.now()
                                    sighting.sms_recipients = [r['phone'] for r in sms_result['results'] if r['success']]
                                    sighting.sms_response = sms_result
                                    sighting.save()
                                    
                                    sms_alerts_sent += sms_result['total_sent']
                                    print(f"SMS Alert sent successfully: {sms_result['total_sent']} recipients")
                                    
                                else:
                                    print(f"SMS Alert failed: {sms_result.get('error', 'Unknown error')}")
                                    sighting.sms_response = sms_result
                                    sighting.save()
                            
                            except Exception as sms_error:
                                print(f"SMS service error: {sms_error}")
                                sighting.sms_response = {'error': str(sms_error)}
                                sighting.save()
                    
                    except Exception as e:
                        print(f"Error processing sighting for {criminal.full_name}: {e}")
            
            # Enhanced success messages
            if len(matches_found) == 1:
                match = matches_found[0]
                criminal = match['criminal_obj']
                base_message = f"🚨 CRIMINAL IDENTIFIED: {criminal.full_name} (Status: {criminal.status.upper()}, Confidence: {match['confidence']:.2f})"
                
                if criminal.status != 'released':
                    if sightings_created > 0:
                        base_message += f" | ✅ Sighting recorded"
                    if sms_alerts_sent > 0:
                        base_message += f" | 📱 SMS alerts sent to {sms_alerts_sent} recipients"
                    base_message += " | 🚔 Authorities have been notified!"
                else:
                    base_message += " | ℹ️ This person has been released - no alerts sent"
                
                messages.success(request, base_message)
            else:
                criminal_names = [m['criminal_obj'].full_name for m in matches_found]
                base_message = f"🚨 MULTIPLE CRIMINALS IDENTIFIED: {', '.join(criminal_names)}"
                
                if sightings_created > 0:
                    base_message += f" | ✅ {sightings_created} sightings recorded"
                if sms_alerts_sent > 0:
                    base_message += f" | 📱 {sms_alerts_sent} SMS alerts sent"
                if sightings_created > 0 or sms_alerts_sent > 0:
                    base_message += " | 🚔 Authorities notified!"
                
                messages.success(request, base_message)
        else:
            messages.warning(request, "No criminal matches found in the uploaded image. The person may not be in the system or the image quality may be insufficient.")

    return redirect('identify_criminal')
# Global camera instance
camera = None
camera_active = False

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load known criminals once using enhanced embeddings
        self.known_embeddings = []
        self.known_backup_embeddings = []
        self.known_names = []
        self.known_national_ids = []
        self.known_criminals = {}
        self.embedding_weights = []  # Quality-based weights for embeddings
        self.load_criminals_enhanced()
        
    def load_criminals_enhanced(self):
        """Load all criminal data using pre-processed embeddings for maximum accuracy"""
        criminals = Criminal.objects.filter(
            face_detected=True,
            face_embedding__isnull=False,
            face_quality_score__gt=0.5
        ).order_by('-face_quality_score')
        loaded_count = 0
        fallback_count = 0
        
        for criminal in criminals:
            try:
                # Use pre-processed embeddings if available
                if criminal.face_embedding and criminal.face_detected:
                    # Add primary embedding
                    primary_embedding = np.array(criminal.face_embedding, dtype=np.float32)
                    self.known_embeddings.append(primary_embedding)
                    self.known_names.append(criminal.full_name)
                    self.known_national_ids.append(criminal.national_id)
                    self.known_criminals[criminal.national_id] = criminal
                    self.embedding_weights.append(criminal.face_quality_score or 1.0)
                    
                    # Add backup embeddings for better matching
                    backup_embeddings = []
                    if criminal.face_embeddings_backup:
                        for backup in criminal.face_embeddings_backup:
                            if isinstance(backup, dict) and 'embedding' in backup:
                                backup_embedding = np.array(backup['embedding'], dtype=np.float32)
                                backup_embeddings.append({
                                    'embedding': backup_embedding,
                                    'enhancement': backup.get('enhancement', 'unknown'),
                                    'confidence': backup.get('confidence', 0.5)
                                })
                    
                    self.known_backup_embeddings.append(backup_embeddings)
                    loaded_count += 1
                    print(f"Loaded enhanced profile for {criminal.full_name} (Quality: {criminal.face_quality_score:.3f}, Backups: {len(backup_embeddings)})")
                    
                else:
                    # Fallback: process image in real-time (less accurate but still works)
                    try:
                        img = cv2.imread(criminal.profile_picture.path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            faces = face_app.get(img)
                            if len(faces) > 0:
                                self.known_embeddings.append(faces[0].normed_embedding)
                                self.known_names.append(criminal.full_name)
                                self.known_national_ids.append(criminal.national_id)
                                self.known_criminals[criminal.national_id] = criminal
                                self.known_backup_embeddings.append([])  # No backups
                                self.embedding_weights.append(0.7)  # Lower weight for fallback
                                fallback_count += 1
                                print(f"Loaded fallback profile for {criminal.full_name}")
                    except Exception as e:
                        print(f"Fallback loading failed for {criminal.full_name}: {e}")
                        
            except Exception as e:
                print(f"Error loading criminal {criminal.full_name}: {e}")
        
        print(f"Criminal recognition system loaded: {loaded_count} enhanced profiles, {fallback_count} fallback profiles")
        
        # Load non-recognition-ready criminals for reference
        non_ready_criminals = Criminal.objects.exclude(
            face_detected=True,
            face_embedding__isnull=False,
            face_quality_score__gt=0.5
        ).count()
        if non_ready_criminals > 0:
            print(f"Warning: {non_ready_criminals} criminal profiles are not ready for recognition (poor image quality or no face detected)")

    def __del__(self):
        if hasattr(self, 'video'):
            self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        if not success:
            return None, []
            
        # Process the frame for face detection
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb_frame)
        
        detection_results = []
        
        for face in faces:
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            embedding = face.normed_embedding
            
            name = "Unknown"
            status = "Unknown"
            confidence = 0.0
            best_match_info = None
            
            if self.known_embeddings:
                # Enhanced matching using multiple embeddings and weighted scoring
                best_match_info = self.enhanced_face_matching(embedding)
                
                if best_match_info and best_match_info['confidence'] > 0.4:  # Adjusted threshold
                    national_id = best_match_info['national_id']
                    criminal_obj = self.known_criminals[national_id]
                    name = criminal_obj.full_name
                    status = criminal_obj.status
                    confidence = best_match_info['confidence']
                    
                    # Record sighting if not released
                    if criminal_obj.status != 'released':
                        try:
                            # Check if this criminal was spotted recently (within last 5 minutes)
                            recent_spotting = CriminalLastSpotted.objects.filter(
                                national_id=criminal_obj.national_id,
                                timestamp__gte=timezone.now() - timedelta(minutes=5)
                            ).exists()
                            
                            if not recent_spotting:
                                # Get camera location
                                location_data = get_camera_location()
                                
                                # Save captured frame as evidence
                                captured_image = save_captured_image(image, criminal_obj.full_name,
                                                                   f"{criminal_obj.national_id}_{int(timezone.now().timestamp())}")
                                
                                # Create sighting record
                                sighting = CriminalLastSpotted.objects.create(
                                    criminal=criminal_obj,
                                    latitude=location_data['latitude'],
                                    longitude=location_data['longitude'],
                                    spotted_address=location_data['address'],
                                    detection_confidence=confidence,
                                    camera_location=location_data['description'],
                                    detection_method='realtime_webcam'
                                )
                                
                                # Save captured image to the sighting
                                if captured_image:
                                    sighting.image_captured = captured_image
                                    sighting.save()
                                
                                print(f"Recorded sighting: {criminal_obj.full_name} (Confidence: {confidence:.3f})")
                                
                                # Send SMS Alert via Arkesel
                                try:
                                    sms_service = ArkeselSMSService()
                                    
                                    # Create image URL if captured image exists
                                    image_url = None
                                    if sighting.image_captured:
                                        image_url = f"{settings.MEDIA_URL}{sighting.image_captured.name}"
                                        if not image_url.startswith('http'):
                                            # Add domain for absolute URL (adjust as needed)
                                            image_url = f"http://localhost:8000{image_url}"
                                    
                                    # Send SMS alert
                                    sms_result = sms_service.send_criminal_alert(criminal_obj, sighting, image_url)
                                    
                                    if sms_result['success']:
                                        # Update sighting with SMS info
                                        sighting.sms_alert_sent = True
                                        sighting.sms_sent_at = timezone.now()
                                        sighting.sms_recipients = [r['phone'] for r in sms_result['results'] if r['success']]
                                        sighting.sms_response = sms_result
                                        sighting.save()
                                        
                                        print(f"SMS Alert sent successfully: {sms_result['total_sent']} recipients")
                                        
                                        # Store success info for UI feedback
                                        best_match_info['sms_sent'] = True
                                        best_match_info['sms_count'] = sms_result['total_sent']
                                    else:
                                        print(f"SMS Alert failed: {sms_result.get('error', 'Unknown error')}")
                                        sighting.sms_response = sms_result
                                        sighting.save()
                                        
                                        best_match_info['sms_sent'] = False
                                        best_match_info['sms_error'] = sms_result.get('error', 'SMS failed')
                                
                                except Exception as sms_error:
                                    print(f"SMS service error: {sms_error}")
                                    sighting.sms_response = {'error': str(sms_error)}
                                    sighting.save()
                                    
                                    best_match_info['sms_sent'] = False
                                    best_match_info['sms_error'] = str(sms_error)
                                
                        except Exception as e:
                            print(f"Error recording sighting: {e}")
            
            # Draw bounding box and label
            color = (0, 255, 0) if status == "released" else (0, 0, 255)
            if confidence > 0.8:
                color = (255, 0, 0)  # High confidence = red
            elif confidence > 0.6:
                color = (255, 165, 0)  # Medium confidence = orange
            
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            
            # Create label with name and status
            if status != "Unknown":
                label = f"{name} - {status.upper()}"
                if confidence > 0:
                    label += f" ({confidence:.2f})"
                if best_match_info and 'match_type' in best_match_info:
                    label += f" [{best_match_info['match_type']}]"
            else:
                label = "Unknown Person"
            
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
            national_id = self.known_national_ids[idx]
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
            criminal_obj = self.known_criminals[national_id]
            
            # Bonus for high-quality profiles
            if hasattr(criminal_obj, 'face_quality_score') and criminal_obj.face_quality_score:
                quality_bonus = criminal_obj.face_quality_score * 0.1
                max_confidence += quality_bonus
            
            # Status-based weight (wanted criminals get slight priority)
            if criminal_obj.status == 'wanted':
                max_confidence *= 1.05
            elif criminal_obj.status == 'under_investigation':
                max_confidence *= 1.02
            
            if max_confidence > best_confidence:
                best_confidence = max_confidence
                best_match = {
                    'national_id': national_id,
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

def gen_frames():
    global camera
    while camera_active and camera:
        frame_data = camera.get_frame()
        if frame_data[0] is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data[0] + b'\r\n')
        time.sleep(0.1)  # Control frame rate

def video_feed(request):
    """Video streaming endpoint"""
    global camera, camera_active
    
    if not camera_active:
        return HttpResponse("Camera not active", status=400)
        
    return StreamingHttpResponse(gen_frames(),
                               content_type='multipart/x-mixed-replace; boundary=frame')

def webcam_stream(request):
    """Render the webcam streaming page"""
    global camera, camera_active
    
    # Initialize camera if not already active
    if not camera_active:
        try:
            camera = VideoCamera()
            camera_active = True
        except Exception as e:
            messages.error(request, f"Could not access webcam: {str(e)}")
            return redirect('identify_criminal')
    
    context = {
        'user': User.objects.get(id=request.session['id']) if request.session.get('id') else None
    }
    return render(request, 'home/webcam_stream.html', context)

def stop_webcam(request):
    """Stop the webcam stream"""
    global camera, camera_active
    
    camera_active = False
    if camera:
        del camera
        camera = None
    
    messages.success(request, "Webcam session ended successfully.")
    return redirect('identify_criminal')







# View to detect criminals using webcam (Updated to use streaming)
def detect_with_webcam(request):
    """Redirect to the new streaming webcam interface"""
    return redirect('webcam_stream')




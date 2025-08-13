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

from django.contrib.auth import get_user_model

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse




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
    sightings_by_day = sightings.extra({'day': "date(timestamp)"}).values('day').annotate(count=Count('id'))
    sightings_dates = [x['day'].strftime('%b %d') for x in sightings_by_day]
    sightings_counts = [x['count'] for x in sightings_by_day]

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
            messages.error(request, "Citizen with that Aadhar Number already exists.")
            return redirect('add_citizen')  # Use named URLs for clarity
        if not profile_picture:
            messages.error(request, "Image is required.")
            return redirect('add_citizen')

        fs = FileSystemStorage()
        filename = fs.save(profile_picture.name, profile_picture)
        uploaded_file_url = fs.url(filename)

        Criminal.objects.create(
            full_name=full_name,
            national_id=national_id,
            address=address,
            profile_picture=uploaded_file_url.lstrip('/'),
            status="wanted"
        )

        messages.success(request, "Citizen successfully added.")
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

        # Load known criminals
        known_embeddings = []
        known_names = []

        for criminal in Criminal.objects.all():
            try:
                img = cv2.imread(criminal.profile_picture.path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = face_app.get(img)
                if len(faces) > 0:
                    embedding = faces[0].normed_embedding
                    known_embeddings.append(embedding)
                    known_names.append(f"{criminal.full_name} ({criminal.address})")
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

        for face in faces:
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            embedding = face.normed_embedding

            name = "Unknown"

            # Compare with known faces
            if known_embeddings:
                distances = []
                for known_emb in known_embeddings:
                    dist = np.linalg.norm(known_emb - embedding)
                    distances.append(dist)
                min_dist = min(distances)
                best_match_idx = np.argmin(distances)

                # Threshold for matching (InsightFace uses ~0.6-0.7 for similarity)
                if min_dist < 0.6:  # Adjust based on your accuracy needs
                    name = known_names[best_match_idx]
                    found = True

            # Draw bounding box and label
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=2)
            try:
                # Use default PIL font or specify a font
                font = ImageFont.load_default()
            except:
                font = None
            text_width, text_height = draw.textsize(name, font=font)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255), font=font)

        del draw

        # Save the result image (optional)
        result_path = uploaded_file_path.replace('.', '_result.')
        pil_image.save(result_path)

        # Show message
        if found:
            messages.success(request, "Criminal match detected successfully!")
        else:
            messages.warning(request, "No match found. The face may not be in the system.")

    return redirect('identify_criminal')





# View to detect criminals using webcam
def detect_with_webcam(request):
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        messages.error(request, "Could not access webcam.")
        return redirect('identify_criminal')

    # Load known criminals (embeddings, names, national IDs)
    known_embeddings = []
    known_names = []
    known_national_ids = []

    criminals = Criminal.objects.all()
    for criminal in criminals:
        try:
            img = cv2.imread(criminal.profile_picture.path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_app.get(img)
            if len(faces) > 0:
                known_embeddings.append(faces[0].normed_embedding)
                full_info = f"Name: {criminal.full_name}, AadharNo: {criminal.national_id}"
                known_names.append(full_info)
                known_national_ids.append(criminal.national_id)
        except Exception as e:
            print(f"Error loading criminal {criminal.full_name}: {e}")

    print("Webcam active. Press 'q' to quit.")  # Console message only

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb_frame)

        for face in faces:
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            embedding = face.normed_embedding

            name = "Unknown"

            if known_embeddings:
                distances = [np.linalg.norm(emb - embedding) for emb in known_embeddings]
                min_dist = min(distances)
                best_idx = np.argmin(distances)

                if min_dist < 0.6:
                    national_id = known_national_ids[best_idx]
                    criminal_obj = Criminal.objects.get(national_id=national_id)
                    name = f"{criminal_obj.full_name} - {criminal_obj.status}"

                    if criminal_obj.status != 'released':
                        CriminalLastSpotted.objects.create(
                            name=criminal_obj.full_name,
                            national_id=criminal_obj.national_id,
                            address=criminal_obj.address,
                            picture=criminal_obj.profile_picture,
                            status=criminal_obj.status,
                            latitude='25.3176° N',
                            longitude='82.9739° E'
                        )

            # Draw rectangle and text on frame (optional, for debug)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # ❌ Remove this line to avoid GUI error
        # cv2.imshow('Video', frame)

        # Check for 'q' keypress using waitKey
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()  # Safe to call even if no window was created

    messages.info(request, "Webcam session ended.")
    return redirect('identify_criminal')




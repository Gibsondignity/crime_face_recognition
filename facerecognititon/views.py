from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, HttpResponse, redirect, get_object_or_404
from django.contrib import messages
import bcrypt
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import cv2
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

        # Load criminals
        known_face_encodings = []
        known_face_names = []

        for criminal in Criminal.objects.all():
            try:
                image = face_recognition.load_image_file(criminal.profile_picture.path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    known_face_encodings.append(encoding[0])
                    known_face_names.append(f"{criminal.full_name} ({criminal.address})")
            except Exception as e:
                print(f"Error loading criminal face: {e}")
                continue

        unknown_image = face_recognition.load_image_file(uploaded_file_path)
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

        pil_image = Image.fromarray(unknown_image)
        draw = ImageDraw.Draw(pil_image)

        found = False  # Track if any match was found

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    found = True

            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        del draw
        pil_image.show()

        # Show message
        if found:
            messages.success(request, "Criminal match detected successfully!")
        else:
            messages.warning(request, "No match found. The face may not be in the system.")

    return redirect('identify_criminal')





# View to detect criminals using webcam
def detect_with_webcam(request):
    # Accessing the deafult camera of the system
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
            return HttpResponse("Webcam not accessible in Docker container", status=500)

    # Loading faces from DB with their data.
    images=[]
    encodings=[]
    names=[]
    files=[]
    nationalIds=[]

    prsn=Criminal.objects.all()
    for criminal in prsn:
        images.append(criminal.full_name+'_image')
        encodings.append(criminal.full_name+'_face_encoding')
        files.append(criminal.profile_picture.path)
        names.append('Name: '+criminal.full_name+ ', AadharNo: '+ criminal.national_id+', Address '+criminal.address)
        nationalIds.append(criminal.national_id)

    #finding encoding of the criminals
    for i in range(0,len(images)):
        images[i]=face_recognition.load_image_file(files[i])
        encodings[i]=face_recognition.face_encodings(images[i])[0]


    # Encoding of faces and their respective ids and names
    known_face_encodings = encodings
    known_face_names = names
    n_id = nationalIds



    while True:
        # Reading a single frame of the video
        ret, frame = video_capture.read()

        # converting color channel from RBG to BRG 
        rgb_frame = frame[:, :, ::-1]

        # Finding all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Run a loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
          
           # checking if the faces in the frame matches to that from our DB
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # finding distance of the faces in the frame to that from our DB
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            #if it matches with the one with minimum distance then print their name on the frame
            if matches[best_match_index]:
                ntnl_id = n_id[best_match_index]
                criminal = Criminal.objects.filter(national_id=ntnl_id)
                name = known_face_names[best_match_index]+', Status: '+criminal.get().status


                # if the face is of a wanted criminal then add it to CriminalLastSpotted list
                if(not(criminal.get().status=='released')):
                    thief = CriminalLastSpotted.objects.create(
                        name=criminal.get().full_name,
                        national_id=criminal.get().national_id,
                        address=criminal.get().address,
                        picture=criminal.get().profile_picture,
                        status=criminal.get().status,
                        latitude='25.3176° N',
                        longitude='82.9739° E'
                    )
                    thief.save()



            # Drawing Rectangular box around the face(s)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Put a label of their name 
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Now display their faces with frames
        cv2.imshow('Video', frame)

        # To quit the webcam detect enter 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Now release the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return redirect('identify_criminal')




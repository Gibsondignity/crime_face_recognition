from __future__ import unicode_literals
from django.db import models
from django.utils import timezone
from datetime import timedelta
import numpy as np

from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.db import models

class CustomUserManager(BaseUserManager):
    use_in_migrations = True

    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("The Email must be set")
        email = self.normalize_email(email)
        extra_fields.setdefault('username', '')  # optional
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields.setdefault('is_active', True)
        extra_fields.setdefault('username', '')  # prevent error

        if extra_fields.get('is_staff') is not True:
            raise ValueError("Superuser must have is_staff=True.")
        if extra_fields.get('is_superuser') is not True:
            raise ValueError("Superuser must have is_superuser=True.")

        return self.create_user(email, password, **extra_fields)


class User(AbstractUser):
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=150, unique=False, blank=True, null=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']  # username is excluded

    objects = CustomUserManager()

    def __str__(self):
        return f"{self.first_name} {self.last_name}"


class AuthorizedPerson(models.Model):
    name = models.CharField(max_length=255, help_text="Full name of the authorized person")
    id_number = models.CharField(max_length=100, unique=True, help_text="Student ID or Staff ID")
    department = models.CharField(max_length=255, help_text="Department or faculty")
    role = models.CharField(max_length=50, choices=[
        ('student', 'Student'),
        ('staff', 'Staff'),
        ('lecturer', 'Lecturer'),
        ('admin', 'Administrator'),
    ], default='student')
    profile_picture = models.ImageField(upload_to='authorized/', null=True, blank=True)
    is_active = models.BooleanField(default=True, help_text="Whether this person is currently authorized")

    # Enhanced face recognition fields
    face_embedding = models.JSONField(null=True, blank=True, help_text="Primary face embedding vector")
    face_embeddings_backup = models.JSONField(null=True, blank=True, help_text="Additional face embeddings for better accuracy")
    face_quality_score = models.FloatField(null=True, blank=True, help_text="Quality score of the primary face (0-1)")
    face_detected = models.BooleanField(default=False, help_text="Whether a face was successfully detected in the image")
    face_detection_confidence = models.FloatField(null=True, blank=True, help_text="Confidence of face detection")
    image_processed = models.BooleanField(default=False, help_text="Whether the image has been processed for face recognition")
    processing_notes = models.TextField(blank=True, help_text="Notes about face processing")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} - {self.id_number} ({self.role})"

    @property
    def recognition_ready(self):
        """Check if this authorized person profile is ready for face recognition"""
        return self.face_detected and self.face_embedding and self.face_quality_score and self.face_quality_score > 0.5
    


class UnauthorizedDetection(models.Model):
    # Since it's unauthorized, no foreign key to person
    detected_name = models.CharField(max_length=255, null=True, blank=True, help_text="Name if identified, otherwise Unknown")
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    detection_address = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    image_captured = models.ImageField(upload_to='detections/', null=True, blank=True)

    # SMS Alert fields
    sms_alert_sent = models.BooleanField(default=False, help_text="Whether SMS alert has been sent")
    sms_sent_at = models.DateTimeField(null=True, blank=True, help_text="When SMS alert was sent")
    sms_recipients = models.JSONField(null=True, blank=True, help_text="List of phone numbers that received alerts")
    sms_response = models.JSONField(null=True, blank=True, help_text="SMS gateway response")

    # Detection details
    detection_confidence = models.FloatField(null=True, blank=True, help_text="Confidence score of the detection")
    camera_location = models.CharField(max_length=255, null=True, blank=True, help_text="Description of camera location")
    detection_method = models.CharField(max_length=50, default='realtime_webcam', help_text="Method used for detection")
    access_attempted = models.CharField(max_length=100, default='lecture_hall', help_text="Area where access was attempted")
    handled = models.BooleanField(default=False, help_text="Whether this detection has been handled")

    # Duplicate prevention
    face_embedding = models.JSONField(null=True, blank=True, help_text="Face embedding for duplicate detection")
    duplicate_of = models.ForeignKey('self', null=True, blank=True, on_delete=models.SET_NULL, help_text="Reference to original detection if this is a duplicate")

    def __str__(self):
        name = self.detected_name or "Unknown Person"
        return f"{name} detected on {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

    @property
    def google_maps_link(self):
        """Generate Google Maps link for the detection location"""
        return f"https://maps.google.com/maps?q={self.latitude},{self.longitude}"

    @property
    def is_duplicate(self):
        """Check if this detection is marked as a duplicate"""
        return self.duplicate_of is not None

    def mark_as_duplicate(self, original_detection):
        """Mark this detection as a duplicate of another"""
        self.duplicate_of = original_detection
        self.save()

    def check_duplicate(self, embedding, threshold=0.8, time_window_minutes=30):
        """
        Check if this face embedding is a duplicate of recent detections
        Returns the original detection if duplicate found, None otherwise
        """
        if not embedding:
            return None

        # Check recent detections within time window
        recent_detections = UnauthorizedDetection.objects.filter(
            timestamp__gte=timezone.now() - timedelta(minutes=time_window_minutes),
            duplicate_of__isnull=True,  # Only check against original detections
            face_embedding__isnull=False
        ).exclude(id=self.id)

        for detection in recent_detections:
            if detection.face_embedding:
                try:
                    # Calculate similarity between embeddings
                    distance = np.linalg.norm(
                        np.array(embedding) - np.array(detection.face_embedding)
                    )
                    similarity = 1.0 - min(distance, 1.0)

                    if similarity >= threshold:
                        print(f"🔄 Duplicate detected: {similarity:.3f} similarity with detection #{detection.id}")
                        return detection
                except (ValueError, TypeError) as e:
                    print(f"⚠️ Error comparing embeddings: {e}")
                    continue

        return None



class SMSConfiguration(models.Model):
    """Arkesel SMS Gateway Configuration"""
    name = models.CharField(max_length=100, default="Main SMS Config")
    api_key = models.CharField(max_length=255, help_text="Arkesel API Key")
    sender_id = models.CharField(max_length=11, help_text="Sender ID (max 11 characters)")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Alert settings
    send_alerts_enabled = models.BooleanField(default=True, help_text="Enable SMS alerts for unauthorized access detection")
    alert_delay_minutes = models.IntegerField(default=5, help_text="Minutes to wait before sending duplicate alerts")
    include_location = models.BooleanField(default=True, help_text="Include location in SMS alerts")
    include_image_link = models.BooleanField(default=True, help_text="Include image link in SMS alerts")

    def __str__(self):
        return f"SMS Config: {self.name}"

    class Meta:
        verbose_name = "SMS Configuration"
        verbose_name_plural = "SMS Configurations"


class AlertRecipient(models.Model):
    """Phone numbers to receive unauthorized access alerts"""
    name = models.CharField(max_length=100, help_text="Name of the recipient")
    phone_number = models.CharField(max_length=15, help_text="Phone number in international format (e.g., +233123456789)")
    department = models.CharField(max_length=100, blank=True, help_text="Department or unit")
    is_active = models.BooleanField(default=True, help_text="Receive alerts")
    priority_level = models.CharField(max_length=20, choices=[
        ('high', 'High Priority - Immediate alerts'),
        ('medium', 'Medium Priority - Standard alerts'),
        ('low', 'Low Priority - Summary alerts only'),
    ], default='medium')

    # Alert preferences - simplified for unauthorized access
    alert_for_unauthorized = models.BooleanField(default=True, help_text="Receive alerts for unauthorized access attempts")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.phone_number})"

    class Meta:
        verbose_name = "Alert Recipient"
        verbose_name_plural = "Alert Recipients"




class File(models.Model):
    file = models.FileField(upload_to='evidence/')
    remark = models.CharField(max_length=100)
    uploaded_by = models.ForeignKey('User', on_delete=models.SET_NULL, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"File: {self.remark} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
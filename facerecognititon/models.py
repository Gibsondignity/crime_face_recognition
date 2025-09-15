from __future__ import unicode_literals
from django.db import models

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


class Criminal(models.Model):
    full_name = models.CharField(max_length=255, help_text="Full name of the criminal")
    national_id = models.CharField(max_length=100, unique=True)    # unique=True
    address = models.TextField()
    profile_picture = models.ImageField(upload_to='criminals/', null=True, blank=True)  # Better than CharField
    status = models.CharField(max_length=50, choices=[
        ('wanted', 'Wanted'),
        ('arrested', 'Arrested'),
        ('under_investigation', 'Under Investigation'),
        ('released', 'Released'),
    ], default='wanted')
    
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
        return f"{self.full_name} - {self.national_id}"
    
    @property
    def recognition_ready(self):
        """Check if this criminal profile is ready for face recognition"""
        return self.face_detected and self.face_embedding and self.face_quality_score and self.face_quality_score > 0.5
    


class CriminalLastSpotted(models.Model):
    criminal = models.ForeignKey(Criminal, on_delete=models.CASCADE, related_name="sightings")
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    spotted_address = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    image_captured = models.ImageField(upload_to='sightings/', null=True, blank=True)
    
    # SMS Alert fields
    sms_alert_sent = models.BooleanField(default=False, help_text="Whether SMS alert has been sent")
    sms_sent_at = models.DateTimeField(null=True, blank=True, help_text="When SMS alert was sent")
    sms_recipients = models.JSONField(null=True, blank=True, help_text="List of phone numbers that received alerts")
    sms_response = models.JSONField(null=True, blank=True, help_text="SMS gateway response")
    
    # Detection details
    detection_confidence = models.FloatField(null=True, blank=True, help_text="Confidence score of the detection")
    camera_location = models.CharField(max_length=255, null=True, blank=True, help_text="Description of camera location")
    detection_method = models.CharField(max_length=50, default='realtime_webcam', help_text="Method used for detection")

    def __str__(self):
        return f"{self.criminal.full_name} spotted on {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def google_maps_link(self):
        """Generate Google Maps link for the spotted location"""
        return f"https://maps.google.com/maps?q={self.latitude},{self.longitude}"



class SMSConfiguration(models.Model):
    """Arkesel SMS Gateway Configuration"""
    name = models.CharField(max_length=100, default="Main SMS Config")
    api_key = models.CharField(max_length=255, help_text="Arkesel API Key")
    sender_id = models.CharField(max_length=11, help_text="Sender ID (max 11 characters)")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Alert settings
    send_alerts_enabled = models.BooleanField(default=True, help_text="Enable SMS alerts for criminal detection")
    alert_delay_minutes = models.IntegerField(default=5, help_text="Minutes to wait before sending duplicate alerts")
    include_location = models.BooleanField(default=True, help_text="Include location in SMS alerts")
    include_image_link = models.BooleanField(default=True, help_text="Include image link in SMS alerts")
    
    def __str__(self):
        return f"SMS Config: {self.name}"
    
    class Meta:
        verbose_name = "SMS Configuration"
        verbose_name_plural = "SMS Configurations"


class AlertRecipient(models.Model):
    """Phone numbers to receive criminal detection alerts"""
    name = models.CharField(max_length=100, help_text="Name of the recipient")
    phone_number = models.CharField(max_length=15, help_text="Phone number in international format (e.g., +233123456789)")
    department = models.CharField(max_length=100, blank=True, help_text="Department or unit")
    is_active = models.BooleanField(default=True, help_text="Receive alerts")
    priority_level = models.CharField(max_length=20, choices=[
        ('high', 'High Priority - Immediate alerts'),
        ('medium', 'Medium Priority - Standard alerts'),
        ('low', 'Low Priority - Summary alerts only'),
    ], default='medium')
    
    # Alert preferences
    alert_for_wanted = models.BooleanField(default=True, help_text="Receive alerts for wanted criminals")
    alert_for_investigation = models.BooleanField(default=True, help_text="Receive alerts for under investigation")
    alert_for_escaped = models.BooleanField(default=True, help_text="Receive alerts for escaped criminals")
    
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
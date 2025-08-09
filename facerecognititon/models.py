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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.full_name} - {self.national_id}"
    


class CriminalLastSpotted(models.Model):
    criminal = models.ForeignKey(Criminal, on_delete=models.CASCADE, related_name="sightings")
    latitude = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    spotted_address = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    image_captured = models.ImageField(upload_to='sightings/', null=True, blank=True)

    def __str__(self):
        return f"{self.criminal.full_name} spotted on {self.timestamp.strftime('%Y-%m-%d %H:%M')}"



class File(models.Model):
    file = models.FileField(upload_to='evidence/')
    remark = models.CharField(max_length=100)
    uploaded_by = models.ForeignKey('User', on_delete=models.SET_NULL, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"File: {self.remark} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
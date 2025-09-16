from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views
from .views import FileView


urlpatterns = [
    path('', views.login_view, name='login'),
    path('dashboard', views.dashboard, name='dashboard'),

    path('add_user', views.add_user, name='add_user'),
    path('save_user', views.save_user, name='save_user'),
    path('view_users', views.view_users, name='view_users'),
    path('change_user_status', views.change_user_status, name='change_user_status'),
    
    path('logout', views.logOut),

    path('detect_unauthorized', views.detect_unauthorized, name='detect_unauthorized'),
    path('detect_image', views.detect_image, name='detect_image'),
    path('detect_with_webcam', views.detect_with_webcam, name='detect_with_webcam'),
    path('webcam_stream', views.webcam_stream, name='webcam_stream'),
    path('stop_webcam', views.stop_webcam, name='stop_webcam'),
    path('upload', FileView.as_view(), name='file-upload'),

    path('spotted_unauthorized', views.spotted_unauthorized, name='spotted_unauthorized'),
    path('mark_detection_handled/<int:detection_id>/', views.mark_detection_handled, name='mark_detection_handled'),
    path('test_detection', views.test_detection_system, name='test_detection_system'),
    path('api/recent_detections', views.get_recent_detections, name='get_recent_detections'),
    path('api/process_frame', views.process_frame, name='process_frame'),

    
    # SMS Configuration URLs
    path('sms_config', views.sms_configuration, name='sms_configuration'),
    path('save_sms_config', views.save_sms_configuration, name='save_sms_configuration'),
    path('test_sms', views.test_sms_service, name='test_sms_service'),
    path('alert_recipients', views.alert_recipients, name='alert_recipients'),
    path('save_alert_recipient', views.save_alert_recipient, name='save_alert_recipient'),
    path('delete_recipient/<int:recipient_id>/', views.delete_alert_recipient, name='delete_alert_recipient'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

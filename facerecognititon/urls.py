from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from . import views
from .views import FileView


urlpatterns = [
    path('', views.login_view, name='login'),
    path('dashboard', views.dashboard, name='dashboard'),

    path('add_criminal', views.add_criminal, name='add_criminal'),
    path('save_criminal', views.save_criminal, name='save_criminal'),
    path('view_criminals', views.view_criminals, name='view_criminals'),
    path('change_status', views.change_criminal_status, name='change_criminal_status'),
    
    path('logout', views.logOut),

    path('identify_criminal', views.identify_criminal, name='identify_criminal'),
    path('detect_image', views.detect_image, name='detect_image'),
    path('detect_with_webcam', views.detect_with_webcam, name='detect_with_webcam'),
    path('upload', FileView.as_view(), name='file-upload'),

    path('spotted_criminal', views.spotted_criminal, name='spotted_criminal'),
    path('found_criminal/<int:thief_id>/', views.found_criminal, name='found_criminal'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

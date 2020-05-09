from django.conf.urls import include, url
from rest_framework import routers
from api import views
from django.conf import settings
from django.conf.urls.static import static

route = routers.DefaultRouter()


urlpatterns = [
    url('api/', include(route.urls)),
    url('api/explain', views.Explain.as_view()),
    url('api/saveFile', views.SaveFile.as_view()),
]

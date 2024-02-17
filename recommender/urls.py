from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('<int:news_id>/', views.news_detail, name='news_detail'),
]

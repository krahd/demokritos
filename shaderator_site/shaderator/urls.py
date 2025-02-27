from django.urls import path

from . import views
from .views import shader_view, store_shader_view, run_management_command, search_shader_descriptions, download_corpus
# for metrics
from .views import store_shader_record

urlpatterns = [
    path('', views.index, name='index'),
    path('shader/', shader_view, name='shader_view'),
    path('store_shader/', store_shader_view, name='store_shader_view'),
    path('store_shader_record/', store_shader_record, name='store_shader_record'),
    path('run-command/', run_management_command, name='run_command'),
    path('search_shader_descriptions/', search_shader_descriptions, name='search_shader_descriptions'),
    path('download_corpus/', download_corpus, name='download_corpus'),
    path('home/', views.home, name='home'),
    path('mobileModal', views.mobile_modal, name='mobile-modal'),
    path('shaders', views.shaders, name='shaders'),
    path('storeInput', views.store_input, name='store-input'),
    path('storeUserInfo', views.store_user_info, name='store-user-info')
]
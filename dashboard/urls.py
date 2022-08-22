from django.conf.urls import url, include
from . import views
from . import accounts as accountSeg 
from . import users as uSeg
from . import conversion as con
from . import promotions as promotionSeg

urlpatterns = [
    url(r'^index/', views.index, name='index'),
    url(r'^account', accountSeg.accountSegmentData, name='account',),
    url(r'^user', uSeg.userSegmentData, name='user',),
    url(r'^conversion', con.conversionSegmentData, name='conversion'),
    url(r'^promotions', promotionSeg.promotionSegmentData, name='promotions'),
    url(r'^cohorts', uSeg.cohorts, name="cohorts"),
]
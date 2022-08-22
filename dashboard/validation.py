from django.shortcuts import render
import datetime
from datetime import date, timedelta


def formDateValidation(request,sDate,eDate,url):

    if not len(sDate):
        return render(request,url, {'flag': '1', 'formValidation': 'Select Start Date'})

    if not len(eDate):
        return render(request,url, {'flag': '1', 'formValidation': 'Select End Date'})
    
    # if not len(dropdown):
    #     return render(request,url, {'flag': '1', 'formValidation' : 'Select a Category'})

    dateTime_sDate = datetime.datetime.strptime(sDate, '%Y-%m-%d')
    dateTime_eDate = datetime.datetime.strptime(eDate, '%Y-%m-%d')

    if (dateTime_eDate - dateTime_sDate).days < 0:
        return render(request,url, {'flag': '1', 'formValidation': 'Please Select A Valid Date Range'})

    return 'valid'
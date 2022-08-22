import datetime
# import SqlDatabase
from django.shortcuts import render
from django import forms as form
from . import mysqlDb as sql
from . import validation as val
from random import sample
from graphos.sources.simple import SimpleDataSource
from graphos.renderers.gchart import PieChart, LineChart, GaugeChart, BarChart


def daterange(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + datetime.timedelta(n)

def promotionSegmentData(request):

    if request.POST.get('promotionsSegmentsubmit') == 'promotionsSegmentform':

        form_sdate = (request.POST.get('sdate'))
        form_edate = (request.POST.get('edate'))

        dropDownValue = (request.POST.get('dropdown'))

        datecheck = val.formDateValidation(request, form_sdate, form_edate, 'dashboard/promotions.html')
        if datecheck != 'valid':
            return datecheck
        
        sdate = form_sdate
        edate = (datetime.datetime.strptime(form_edate, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime(
            '%Y-%m-%d')
        sdate = datetime.datetime.strptime(sdate, '%Y-%m-%d').strftime('%b %d, %Y')
        edate = datetime.datetime.strptime(edate, '%Y-%m-%d').strftime('%b %d, %Y')

        print(form_sdate, form_edate)
        #sorting date for the graph
        datasort_sDate = datetime.datetime.strptime(form_sdate, '%Y-%m-%d')
        datasort_eDate = datetime.datetime.strptime(form_edate, '%Y-%m-%d')

        if dropDownValue == 'Segment1' or dropDownValue == 'Segment2' or dropDownValue == 'Segment3':

            segment1 = sql.getData("SELECT count(id) FROM users WHERE is_blocked != 1 AND id IN(SELECT user_coins_log.user_id FROM user_coins_log \
                    WHERE user_coins_log.user_id IN (SELECT users.id FROM users WHERE date(users.created_at)>='2020-01-01') GROUP BY user_coins_log.user_id HAVING \
                    (sum(case when user_coins_log.source_type = 'ghoori-bkash' then 1 else 0 end)>0 AND MAX(date(user_coins_log.created_at)) < '2021-01-31'))")

            user = 0
            for i in segment1:
                user = i[0]
        # promoUsedUser = sql.getData("select p.*, sg.group_name,(select count(id) from user_promo where promo_id = p.id) as promo_used_count \
        #                         from promo as p inner join segment_groups as sg on p.segment_group_id = sg.id order by p.validity desc")
            #percentage = (user / promoUsedUser) * 100
        
        coinPackagePromotions = sql.getData("SELECT user_id,count(*) as cnt FROM user_coins_log where recharge_amount=28 and date(created_at) between '2021-04-14' and '2021-04-14' group by user_id order by cnt")
        coinPackagePromotions = [[users, count] for users, count in coinPackagePromotions]
        print(coinPackagePromotions)
        coinPackagePromotions.insert(0, ['Users', 'Count'])
        coinPackagePromotions = BarChart(SimpleDataSource(data=coinPackagePromotions))
        packageTaken = sql.getData("SELECT count(user_id) FROM user_coins_log where recharge_amount=28 and date(created_at) between '2021-04-14' and '2021-04-14'")
        taken = 0
        
        for taken in packageTaken:
            taken = taken[0]
        packageTakenUnique = sql.getData("SELECT count(DISTINCT(user_id)) FROM user_coins_log where recharge_amount=28 and date(created_at) between '2021-04-13' and '2021-04-14'")
        unique = 0
        for i in packageTakenUnique:
            unique = i[0]
        return render(request, 'dashboard/promotions.html', {'segment1': user, 'cpChart': coinPackagePromotions, 'packageTaken': taken, 
                                                            'uniqueUser': unique,'sdate': sdate, 'edate': edate})
    else:
        return render(request, 'dashboard/promotions.html')



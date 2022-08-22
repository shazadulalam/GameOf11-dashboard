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

def accountSegmentData(request):

    if request.POST.get('accountSegmentsubmit') == 'accountSegmentform':

        form_sdate = (request.POST.get('sdate'))
        form_edate = (request.POST.get('edate'))

        # print(form_sdate, form_edate)

        datecheck = val.formDateValidation(request, form_sdate, form_edate, 'dashboard/accounts.html')
        if datecheck != 'valid':
            return datecheck
        
        sdate = form_sdate
        edate = (datetime.datetime.strptime(form_edate, '%Y-%m-%d') + datetime.timedelta(days=1)).strftime(
            '%Y-%m-%d')
        sdate = datetime.datetime.strptime(sdate, '%Y-%m-%d').strftime('%b %d, %Y')
        edate = datetime.datetime.strptime(edate, '%Y-%m-%d').strftime('%b %d, %Y')

        #sorting date for the graph
        datasort_sDate = datetime.datetime.strptime(form_sdate, '%Y-%m-%d')
        datasort_eDate = datetime.datetime.strptime(form_edate, '%Y-%m-%d')

        all_date = [dt.strftime('%Y-%m-%d') for dt in daterange(datasort_sDate, datasort_eDate)]
        sortedDate = sorted(all_date)

        withdraw, outgoingTransaction, dayDue, totalDue = accountTransType(form_sdate, form_edate)
        # print(withdraw)
        withdraw = [[date, amount]for date, amount in withdraw]
        withdraw.insert(0,['Date', 'Amount'])
        withdrawChart = LineChart(SimpleDataSource(data=withdraw))

        #total outgoing transaction
        amount = 0
        for i in outgoingTransaction:
            amount = i[0]

        #daily due findings
        dailyDue = [[day, amount] for day, amount in dayDue]
        dailyDue.insert(0, ['Date', 'Amount'])
        dueChart = PieChart(SimpleDataSource(data=dailyDue))

        #total game of 11 dues
        due = 0
        for i in totalDue:
            due = i[0]


        return render(request, 'dashboard/accounts.html', {'w_chart': withdrawChart, 'total_outgoing': amount, 'dueChart':dueChart,
                                                            'due':due, 'sdate': sdate, 'edate': edate})
    else:
        return render(request, 'dashboard/accounts.html')


def accountTransType(sdate, edate):

    withdraw = sql.getData("SELECT cast(created_at as date) as date,sum(amount) from user_cash_log where request_type = 'withdraw' \
                                and status = 'success' and created_at between '" + sdate + "' and '"+ edate +"' group by created_at")

    outgoingTransaction = sql.getData("SELECT sum(amount) from user_cash_log where request_type = 'withdraw' \
                                         and status = 'success'")

    dayDue = sql.getData("SELECT cast(created_at as date) as date, sum(amount) from user_cash_log where request_type = 'income' \
                            and created_at BETWEEN '"+ sdate +"' and '"+ edate +"' group by created_at")
    
    totalDue = sql.getData("SELECT sum(amount) from user_cash_log where request_type = 'income'")

    return withdraw, outgoingTransaction, dayDue, totalDue
    
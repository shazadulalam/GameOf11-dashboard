# GameOf11

This is in-house dashboard demo for GameOf11. GameOf11 is the first Daily Fantasy Sports (DFS) platform in Bangladesh. GameOF11 aka GO11 is fantasy sports where users can build fantasy team, enter contests and win cash prizes at the end of every game. So users don't have to wait for the long season to get the prizes.

Our goal is to create best fantasy platform In Bangladesh. We are increasing our coverage areas in terms of Cricket and Football daily matches. 

Right now we are looking for early stage investment to provide better service to our users and increase user's engagement. 

### Prerequisites

Before you start make sure you have everything you need to oversee this repository.

* Python
* pip (updated Version)
* Django Basic

# Analytics Dashboard

### Installing

Django Install 

```
pip install Django
```

And for GRAPHOS

```
pip install django-graphos
```


## Database
To configure the database please use your login credential and use it in MySql.py. If any error occure during the run period please install mysql-client for your windows/linux/mac.

For installing pyodbc:

```
pip install pyodbc
```

### Problem

Accessing MySQL from Django, sometimes create problems with pyodbc and ODBC driver. You need to install boths to run the server.  

* [ODBC Connector](https://dev.mysql.com/downloads/connector/odbc/3.51.html) - Install it from here
* [Ubuntu Installation](https://askubuntu.com/questions/800216/installing-ubuntu-16-04-lts-how-to-install-odbc)- Use this Reference 

If you somehow face any problems while running the dashboard and the problem cant be solved by you then 

```
submit a issues and will be fixed hopefully
```


## Deployment

For running the server, go the directory and in terminal type

```
python manage.py runserver
```


Add additional address parameter after this http://127.0.0.1:8000/

For Accounts:
```
http://127.0.0.1:8000/dashboard/account/
```
and for user Segmentation 
```
http://127.0.0.1:8000/dashboard/user/
```

## License

This project is licensed under the MIT License - all rights reserved by Umbrella Inc.




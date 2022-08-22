from django.test import TestCase
import mysqlDb as m
import pandas as pd


def fetchData(match_id):

    # matchId = data.getData("SELECT matches.id, matches.name, matches.match_time, count(contests.id) as contest_count FROM matches join contests ON matches.id = contests.match_id WHERE matches.is_published = 1 AND matches.match_time >= '2019-03-20' AND matches.match_time < '2020-02-24' GROUP BY matches.id")


    calculateData = m.getData("SELECT SUM(Total_Per_Contest.Total_Entry_Amount_Per_Contest)/50 AS Total_Entry_Amount, SUM(Total_Per_Contest.Total_Seat_Per_Contest) AS Total_Seat, Sum(Total_Per_Contest.winning_amount) AS Total_Winning_Amount, Sum(Total_Per_Contest.teams_capacity) AS Total_Team_Capacity, (SUM(Total_Per_Contest.Total_Entry_Amount_Per_Contest)/50 -Sum(Total_Per_Contest.winning_amount)) As Profit_Mergin, COUNT(Total_Per_Contest.Contest_ID) As Contest_Count FROM (SELECT COUNT(U_T_C.contest_id)*C.entry_amount as Total_Entry_Amount_Per_Contest, COUNT(U_T_C.contest_id) As Total_Seat_Per_Contest, C.winning_amount, C.teams_capacity, C.id AS Contest_ID FROM user_team_contests as U_T_C INNER JOIN (SELECT id, entry_amount, winning_amount,teams_capacity FROM contests WHERE match_id= "+ match_id +" and contest_type='paid') as C on U_T_C.contest_id = C.id GROUP BY U_T_C.contest_id) As Total_Per_Contest")


    return calculateData


# calculateData = data.getData("SELECT SUM(Total_Per_Contest.Total_Entry_Amount_Per_Contest)/50 AS Total_Entry_Amount, SUM(Total_Per_Contest.Total_Seat_Per_Contest) AS Total_Seat, Sum(Total_Per_Contest.winning_amount) AS Total_Winning_Amount, Sum(Total_Per_Contest.teams_capacity) AS Total_Team_Capacity, (SUM(Total_Per_Contest.Total_Entry_Amount_Per_Contest)/50 -Sum(Total_Per_Contest.winning_amount)) As Profit_Mergin, COUNT(Total_Per_Contest.Contest_ID) As Contest_Count FROM (SELECT COUNT(U_T_C.contest_id)*C.entry_amount as Total_Entry_Amount_Per_Contest, COUNT(U_T_C.contest_id) As Total_Seat_Per_Contest, C.winning_amount, C.teams_capacity, C.id AS Contest_ID FROM user_team_contests as U_T_C INNER JOIN (SELECT id, entry_amount, winning_amount,teams_capacity FROM contests WHERE match_id= 131 and contest_type='paid') as C on U_T_C.contest_id = C.id GROUP BY U_T_C.contest_id) As Total_Per_Contest")
matchId = m.getData("SELECT matches.id, matches.name, matches.match_time, count(contests.id) as contest_count FROM matches join contests ON matches.id = contests.match_id WHERE matches.is_published = 1 AND matches.match_time >= '2019-03-20' AND matches.match_time < '2020-02-24' GROUP BY matches.id")

print(matchId)

mId = []
df_processed = []
counted_df = []
for match_id, name, matchTime, contestCount in matchId:

    mid = fetchData(str(match_id))
    mId.append(mid)
    df_processed.append([str(match_id), name, matchTime, contestCount])

    


columns = ['match_id', 'name', 'matchTime', 'total contest given']
df = pd.DataFrame(df_processed, columns=columns)

for data in mId:
    for frequency in data:
        counted_df.append([frequency[0], frequency[1], frequency[2], frequency[3], frequency[4]])

coulumn1 = ['Total Entry Amount', 'TotalSeat', 'TotalWinningAMount', 'TotalTeamCapacity', 'ProfitMergin']
df1 = pd.DataFrame(counted_df, columns=coulumn1)

final_df = pd.concat([df, df1], axis=1)



# Create your tests here.

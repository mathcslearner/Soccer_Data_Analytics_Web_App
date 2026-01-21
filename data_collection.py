from soccerdata import FBref
import pandas as pd

rename_columns = {
    'Performance_Gls': 'Goals',
    'Performance_Ast': 'Assist',
    'Performance_G+A': 'G+A',
    'Performance_G-PK': 'NPG',
    'Performance_PK': 'PK Gls',
    'Performance_PKatt': 'PK Att',
    'Performance_CrdY': 'Yellow Cards',
    'Performance_CrdR': 'Red Cards',

    'Expected_xG': 'xG',
    'Expected_npxG': 'npxG',
    'Expected_xAG': 'xAG',
    'Expected_npxG+xAG': 'npxG+xAG',

    'Progression_PrgC': 'Prg Carries',
    'Progression_PrgP': 'Prg Passes',

    'Standard_Sh': 'Shots',
    'Standard_SoT': 'Shots on Target',
    'Standard_SoT%': 'SoT%',
    'Standard_G/Sh': 'Goals per Shots',
    'Standard_G/SoT': 'Goals per SoT',
    'Standard_Dist': 'Avg Shot Dist',
    'Standard_FK': 'FK Shots',

    'Expected_npxG/Sh': 'npxG/Shot',
    'Expected_G-xG': 'G-xG',
    'Expected_np:G-xG': 'G-npxG',

    'Total_Cmp': 'Passes Completed',
    'Total_Att': 'Passes Attempted',
    'Total_Cmp%': 'Pass Completion%',
    'Total_TotDist': 'Avg Pass Distance',
    'Total_PrgDist': 'Avg Prg Pass Distance',

    'Short_Cmp': 'Short Pass Cmp',
    'Short_Att': 'Shot Pass Att',
    'Short_Cmp%': 'Short Pass Cmp%',

    'Medium_Cmp': 'Medium Pass Cmp',
    'Medium_Att': 'Medium Pass Att',
    'Medium_Cmp%': 'Medium Pass Cmp%',

    'Long_Cmp': 'Long Pass Cmp',
    'Long_Att': 'Long Pass Att',
    'Long_Cmp%': 'Long Pass Cmp%',

    'xAG': 'xA',
    'Expected_A-xAG': 'A-xAG',
    'KP': 'Key Passes',
    '1/3': 'Final Third Pass',
    'PPA': 'Pass into Pen Area',
    'CrsPA': 'Cross into Pen Area',

    'Pass Types_Live': 'Open Play Pass',
    'Pass Types_Dead': 'Dead Ball Pass',
    'Pass Types_FK': 'FK Pass',
    'Pass Types_TB': 'Through Balls',
    'Pass Types_Sw': 'Switches',
    'Pass Types_TI': 'Throw-Ins',
    'Pass Types_CK': 'Corner Kicks',

    'Corner Kicks_In': 'Inswing Corner',
    'Corner Kicks_Out': 'Outswing Corner',
    'Corner Kicks_Str': 'Straight Corner',

    'Outcomes_Blocks': 'Blocked Passes',

    'SCA_SCA': 'SCA',
    'SCA Types_PassLive': 'SCA Open Play',
    'SCA Types_PassDead': 'SCA Dead Ball',
    'SCA Types_TO': 'SCA Take-Ons',
    'SCA Types_Sh': 'Shots SCA',
    'SCA Types_Fld': 'Fouled SCA',
    'SCA Types_Def': 'Defensive SCA',

    'GCA_GCA': 'GCA',
    'GCA Types_PassLive': 'GCA Open Play',
    'GCA Types_PassDead': 'GCA Dead Ball',
    'GCA Types_TO': 'GCA Take-Ons',
    'GCA Types_Sh': 'Shots GCA',
    'GCA Types_Fld': 'Fouled GCA',
    'GCA Types_Def': 'Defensive GCA',

    'Tackles_Tkl': 'Tackles',
    'Tackles_TklW': 'Tackles Won',
    'Tackles_Def 3rd': 'Tackles Def 3rd',
    'Tackles_Mid 3rd': 'Tackles Mid 3rd',
    'Tackles_Att 3rd': 'Tackles Att 3rd',
    'Challenges_Tkl': 'Dribble Stops',
    'Challenges_Att': 'Total Challenges',
    'Challenges_Tkl%': 'Challenges Won%',
    'Challenges_Lost': 'Challenges Lost',

    'Blocks_Blocks': 'Blocks',
    'Blocks_Sh': 'Shot Blocks',
    'Int': 'Interceptions',
    'Tkl+Int': 'Tackle+Interceptions',
    'Clr': 'Clearances',
    'Err': 'Errors',

    'Touches_Touches': 'Touches',
    'Touches_Def Pen': 'Touches Def Pen',
    'Touches_Def 3rd': 'Touches Def 3rd',
    'Touches_Mid 3rd': 'Touches Mid 3rd',
    'Touches_Att 3rd': 'Touches Att 3rd',
    'Touches_Att Pen': 'Touches in Box',
    'Touches_Live': 'Open Play Touches',

    'Take-Ons_Att': 'Take-Ons Attempted',
    'Take-Ons_Succ': 'Take-Ons Successfull',
    'Take-Ons_Succ%': 'Take-Ons Success%',
    'Take-Ons_Tkld': 'Take-Ons Tackled',
    'Take-Ons_Tkld%': 'Take-Ons Tackled%',

    'Carries_Carries': 'Carries',
    'Carries_TotDist': 'Avg Carry Dist',
    'Carries_PrgDist': 'Avg Prg Carry Dist',
    'Carries_1/3': 'Carries into Final Third',
    'Carries_CPA': 'Carries into Pen Area',
    'Carries_Mis': 'Miscontrols',
    'Carries_Dis': 'Dispossessed',

    'Receiving_Rec': 'Passes Received',
    'Receiving_PrgR': 'Progressive Receives',

    'Subs_Mn/Sub': 'Min per Sub',

    'Team Success_PPM': 'Pts pMatch',
    'Team Success_onGA': 'Goals Allowed',
    'Team Success_+/-': 'Goal Difference',
    'Team Success (xG)_onxGA': 'xG Allowed',
    'Team Success (xG)_xG+/-': 'xG Difference',

    'Performance_Fls': 'Fouls',
    'Performance_Fld': 'Fouled',
    'Performance_Off': 'Offside',
    'Performance_Crs': 'Crosses',
    'Performance_PKcon': 'Pen Conceded',
    'Performance_OG': 'Own Goals',
    'Performance_Recov': 'Recoveries',

    'Aerial Duels_Won': 'Aerial Duels Won',
    'Aerial Duels_Lost': 'Aerial Duels Lost',
    'Aerial Duels_Won%': 'Aerial Duels Won%',

    'Performance_SoTA': 'Shots on Target Allowed',
    'Performance_Saves': 'Saves',
    'Performance_Save%': 'Save%',

    'Performance_W': 'W',
    'Performance_D': 'D',
    'Performance_L': 'L',
    'Performance_CS': 'Clean Sheet',
    'Performance_CS%': 'Clean Sheet%',

    'Penalty Kicks_PKsv': 'Penalty Saves',
    'Penalty Kicks_Save%': 'Penalty Save%',

    'Goals_FK': 'FK Goals',
    'Goals_CK': 'Corner Goals',

    'Passes_Att (GK)': 'Pass Attempts(GK)',
    'Passes_Thr': 'Throws',
    'Passes_Launch%': 'Passes Launch%',
    'Passes_AvgLen': 'Passes AvgLen',

    'Goal Kicks_Att': 'Goal Kicks Attempted',
    'Goal Kicks_Launch%': 'Goal Kicks Launch%',
    'Goal Kicks_AvgLen': 'Goal Kicks AvgLen',

    'Crosses_Opp': 'Crosses Faced',
    'Crosses_Stp': 'Crosses Stopped',
    'Crosses_Stp%': 'Cross Stop%',

    'Sweeper_#OPA': 'OPA',
    'Sweeper_AvgDist': 'OPA Avg Dist',
    'Expected_xA': 'xA'
}

def get_data():
    fb = FBref(leagues = "Big 5 European Leagues Combined", seasons = "24-25")

    standard_stats = fb.read_team_season_stats("standard")
    standard_stats = standard_stats.reset_index()
    standard_stats.columns = ["_".join([str(c) for c in col if c]) for col in standard_stats.columns]
    standard_stats = standard_stats.drop(columns = ["Playing Time_MP", "Playing Time_Starts", "Playing Time_Min", "Playing Time_90s", "Per 90 Minutes_Gls", "Per 90 Minutes_Ast",
        "Per 90 Minutes_G+A", "Per 90 Minutes_G-PK", "Per 90 Minutes_G+A-PK", "Per 90 Minutes_xG", "Per 90 Minutes_xAG", "Per 90 Minutes_xG+xAG", "Per 90 Minutes_npxG", "Per 90 Minutes_npxG+xAG", "url"])
    
    shooting_stats = fb.read_team_season_stats("shooting")
    shooting_stats = shooting_stats.reset_index()
    shooting_stats.columns = ["_".join([str(c) for c in col if c]) for col in shooting_stats.columns]
    shooting_stats = shooting_stats.drop(columns = ["Expected_xG", "Expected_npxG", "Standard_Gls", "90s", "Standard_Sh/90", "Standard_SoT/90","Standard_PK", "Standard_PKatt", 
        "Expected_xG", "Expected_npxG", "url", "players_used"])
    
    passing_stats = fb.read_team_season_stats("passing")
    passing_stats = passing_stats.reset_index()
    passing_stats.columns = ["_".join([str(c) for c in col if c]) for col in passing_stats.columns]
    passing_stats = passing_stats.drop(columns = ["90s", "Ast", "PrgP", "url", "players_used", "xAG"])

    passing_types_stats = fb.read_team_season_stats("passing_types")
    passing_types_stats = passing_types_stats.reset_index()
    passing_types_stats.columns = ["_".join([str(c) for c in col if c]) for col in passing_types_stats.columns]
    passing_types_stats = passing_types_stats.drop(columns = ["90s", "url", "Att", "Pass Types_Crs", "Outcomes_Cmp", "Outcomes_Off", "players_used"])

    goal_shot_creation_stats = fb.read_team_season_stats("goal_shot_creation")
    goal_shot_creation_stats = goal_shot_creation_stats.reset_index()
    goal_shot_creation_stats.columns = ["_".join([str(c) for c in col if c]) for col in goal_shot_creation_stats.columns]
    goal_shot_creation_stats = goal_shot_creation_stats.drop(columns = ["90s", "SCA_SCA90", "GCA_GCA90", "url", "players_used"])

    defense_stats = fb.read_team_season_stats("defense")
    defense_stats = defense_stats.reset_index()
    defense_stats.columns = ["_".join([str(c) for c in col if c]) for col in defense_stats.columns]
    defense_stats = defense_stats.drop(columns = ["90s", "Blocks_Pass", "url", "players_used"])

    possession_stats = fb.read_team_season_stats("possession")
    possession_stats = possession_stats.reset_index()
    possession_stats.columns = ["_".join([str(c) for c in col if c]) for col in possession_stats.columns]
    possession_stats = possession_stats.drop(columns = ["Poss", "90s", "url", "Carries_PrgC", "players_used"])

    playing_time_stats = fb.read_team_season_stats("playing_time")
    playing_time_stats = playing_time_stats.reset_index()
    playing_time_stats.columns = ["_".join([str(c) for c in col if c]) for col in playing_time_stats.columns]
    playing_time_stats = playing_time_stats.drop(columns = ["Age", "url", "Playing Time_MP", "Playing Time_Min", "Playing Time_Mn/MP", "Playing Time_Min%", "Playing Time_90s", "Starts_Starts", 
        "Starts_Mn/Start", "Starts_Compl", "Subs_Subs","Subs_unSub","Team Success_onG", "Team Success_+/-90", "Team Success (xG)_onxG", "Team Success (xG)_xG+/-90", "players_used"])
    
    misc_stats = fb.read_team_season_stats("misc")
    misc_stats = misc_stats.reset_index()
    misc_stats.columns = ["_".join([str(c) for c in col if c]) for col in misc_stats.columns]
    misc_stats = misc_stats.drop(columns = ["90s", "Performance_CrdY", "Performance_CrdR", "Performance_2CrdY", "Performance_Int", "Performance_TklW", "Performance_PKwon", "url", "players_used"])

    keeper_stats = fb.read_team_season_stats("keeper")
    keeper_stats = keeper_stats.reset_index()
    keeper_stats.columns = ["_".join([str(c) for c in col if c]) for col in keeper_stats.columns]
    keeper_stats = keeper_stats.drop(columns = ["Playing Time_MP", "Playing Time_Starts", "Playing Time_Min", "90s", "Performance_GA", "Performance_GA90", "Penalty Kicks_PKatt", "Penalty Kicks_PKA", "Penalty Kicks_PKm", "url", "players_used"])

    keeper_adv_stats = fb.read_team_season_stats("keeper_adv")
    keeper_adv_stats = keeper_adv_stats.reset_index()
    keeper_adv_stats.columns = ["_".join([str(c) for c in col if c]) for col in keeper_adv_stats.columns]
    keeper_adv_stats = keeper_adv_stats.drop(columns = ["90s", "url", "Goals_GA", "Goals_PKA", "Goals_OG", "Expected_PSxG", "Expected_PSxG/SoT", "Expected_PSxG+/-", "Expected_/90", "Launched_Cmp", "Launched_Att", "Launched_Cmp%", "Sweeper_#OPA/90", "players_used"])

    data1 = pd.merge(standard_stats, shooting_stats, on = ["league", "season", "team"], how = "inner")
    data2 = pd.merge(passing_stats, passing_types_stats, on = ["league", "season", "team"], how = "inner")
    data3 = pd.merge(goal_shot_creation_stats, defense_stats, on = ["league", "season", "team"], how = "inner")
    data4 = pd.merge(possession_stats, playing_time_stats, on = ["league", "season", "team"], how = "inner")
    data5 = pd.merge(misc_stats, keeper_stats, on = ["league", "season", "team"], how = "inner")
    

    data = pd.merge(data1, data2, on = ["league", "season", "team"], how = "inner")
    data = data.merge(data3, on = ["league", "season", "team"], how = "inner")
    data = data.merge(data4, on = ["league", "season", "team"], how = "inner")
    data = data.merge(data5, on = ["league", "season", "team"], how = "inner")
    data = data.merge(keeper_adv_stats, on = ["league", "season", "team"], how = "inner")

    data = data.rename(columns = rename_columns)

    data["Avg Pass Distance"] = data["Avg Pass Distance"].div(data["Passes Attempted"])
    data["Avg Prg Pass Distance"] = data["Avg Prg Pass Distance"].div(data["Prg Passes"])
    data["Avg Carry Dist"] = data["Avg Carry Dist"].div(data["Carries"])
    data["Avg Prg Carry Dist"] = data["Avg Prg Carry Dist"].div(data["Prg Carries"])

    return data

data = get_data()
data.to_csv("team_stats_2024-2025.csv", index=False)

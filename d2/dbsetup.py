import sqlite3 as lite
import sys
import json
import unirest
import urllib2
import pickle
import time, collections

'''
-------------------------------------
Loads matches into local sqlite
database using Steam API
-------------------------------------
'''

def get_new_match():
    key = '8469FB5F835FCEBE1166E994E756A252'
    ## Test account_id = 133662126
    match_response = urllib2.urlopen('''https://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/V001/?matches_requested=%i&skill=%i&account_id=133662126&key=%s'''%(1, 0, key))
    match = json.load(match_response)['result']['matches'][0]
    m_id = match['match_id']
    response = urllib2.urlopen('https://api.steampowered.com/IDOTA2Match_570/GetMatchDetails/V001/?match_id=%s&key=%s' %(m_id, key))
    d = json.load(response, object_pairs_hook=collections.OrderedDict)['result']
    return d['players']
        
def populate(**kwargs):
    key = '8469FB5F835FCEBE1166E994E756A252'
    query_string = key
    for k in kwargs:
        if kwargs[k]:
            query_string += '&%s=%s' %(k, kwargs[k])
    skill_map = {0: 'any', 1: 'normal', 2: 'high', 3: 'very_high'}
    match_response = urllib2.urlopen('''https://api.steampowered.com/IDOTA2Match_570/GetMatchHistoryBySequenceNum/V001/?key=%s'''%(query_string))
    try:
        matches = json.load(match_response)['result']['matches']
    except KeyError:
        time.sleep(.2)
        populate(**kwargs)

    con = lite.connect('test.db')

    ## If skill group is 'any', sends to generic 'match' table.
    ## Otherwise, creates a skill-specific match table.
    ## The api response for match-history doesn't hold skill group,
    ## so you can't know what group the match is in unless you
    ## explicitly searched for a certain group.
    table_string = ''
    if kwargs['skill'] == '0':
        pass
    else:
        table_string = skill_map[int(kwargs['skill'])] + '_'

    with con:
        cur = con.cursor()
        
        ## Creates table with the following columns
        cur.execute('''CREATE TABLE IF NOT EXISTS %smatches(
                    players BLOB,
                    radiant_win INTEGER,
                    duration INTEGER,
                    start_time INTEGER,
                    match_id INTEGER,
                    match_seq_num INTEGER,
                    tower_status_radiant INTEGER,
                    tower_status_dire INTEGER,
                    barracks_status_radiant INTEGER,
                    barracks_status_dire INTEGER,
                    cluster INTEGER,
                    first_blood_time INTEGER,
                    lobby_type INTEGER,
                    human_players INTEGER,
                    league_id INTEGER,
                    positive_votes INTEGER,
                    negative_votes INTEGER,
                    game_mode INTEGER)''' %(table_string))

        count = 1
        for num, match in enumerate(matches):
            m_id = match['match_id']
            response = urllib2.urlopen('https://api.steampowered.com/IDOTA2Match_570/GetMatchDetails/V001/?match_id=%s&key=%s' %(m_id, key))
            d = json.load(response, object_pairs_hook=collections.OrderedDict)['result']
            time.sleep(.1)
            values = [d[k] for k in d]
                                     
            ## Only want > 10 min games
            ## API can't filter by duration..
            if d['duration'] < 600:
                continue
            if len(values) > 18:
                values = values[:18]
            for i, e in enumerate(values):
                if type(e) == list:
                    values[i] = pickle.dumps(e)
            s = '?,'*(len(values)-1)
            cur.execute('''SELECT match_id from %smatches
                        WHERE match_id = ?''' %(table_string), [m_id])
            if not cur.fetchone():
                cur.execute('INSERT INTO %smatches VALUES (%s?)' %(table_string, s), values)
                #print 'inserted %i matches...' %(count)
                count += 1
        return d['match_seq_num']

        
if __name__ == '__main__':
    ## Starting match sequence number (larger -> more recent)
    msn = 11215

    ## skill 3 - very high
    ## any league
    ## starting at msn
    kwargs = {'matches_requested': '25', 'skill': '3', 'league_id': '',\
              'account_id': '', 'start_at_match_seq_num' : '%i' %(msn)}
    count = 0      
    while True:
        kwargs['start_at_match_seq_num'] = populate(**kwargs) + 1
        print 'Finished batch...'
        time.sleep(1)
        count += 1
        if count > 50:
            break
        print 'Fetching more matches...\n'

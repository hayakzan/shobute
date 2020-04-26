# -*- coding: utf-8 -*-

# hayakzan - Shobute
# a Python-based music notation interface
# works under Python 2.x, needs work for Python 3.x.

# dependencies
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.transforms import blended_transform_factory
from matplotlib.figure import Figure, figaspect
import random
from scipy import interpolate
from scipy.interpolate import spline
from scipy.interpolate import splprep, splev
from pylab import figure, axes, pie, title, show
from matplotlib.lines import Line2D
import matplotlib.patches as mpatch
import csv

plt.cla()

# seed for randomization
#random.seed(0)

# mise-en-page
size = 10
x_off  = 0.4
y_off = 9.2
y_offset_2 = 0
y_off3 = -2.7
yowadd = 1.3
yowfac = 0
fig = plt.figure(figsize=(24, 6))
ax = plt.subplot(1, 1, 1)

ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

# dictionaries
pitch_dict = { 40: -3.5+y_off, 41: -3+y_off, 42: -3+y_off, 43: -2.5+y_off, 44: -2.5+y_off, 45: -2+y_off, 46: -2+y_off, 47: -1.5+y_off, 48: -1+y_off,
               49: -1+y_off, 50: -0.5+y_off, 51: -0.5+y_off, 52: 0+y_off, 53: 0.5+y_off, 54: 0.5+y_off }
yow_dict = { 1.0: yowadd*0+y_off3, 2.0: yowadd*2+y_off3, 3.0: yowadd*3+y_off3, 4.0: yowadd*4+y_off3,
             5.0: yowadd*5+y_off3, 6.0: yowadd*6+y_off3}
oct_dict = { 0: 0, 12: 3.5, 24: 7, 36: 10.5 }

pha_dict = { 1: x_off, 2: 1.235+x_off, 2.5: (1.235+0.6175)+x_off, 3: (1.235*2)+x_off, 4: (1.235*3)+x_off,
                5: (1.235*4)+x_off, 6: (1.235*5)+x_off, 7: (1.235*6)+x_off, 8: (1.235*7)+x_off }
pos_dict = { 'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0, 'e': 5.0, 'f': 6.0, 'g': 7.0, 'h': 8.0, 'j': 9.0, 'k': 10.0, 'l': 11.0,
             'm': 12.0, 'n': 13.0, 'o': 14.0, 'p': 15.0, 'q': 16.0, 'r': 17.0, 's': 18.0, 't': 19.0 }

# fonts
font1 = {'family': 'Opus Special Std',
        'color':  'black',
        'weight': 'normal',
        'size': 22
        }

font2 = {'family': 'Opus Std',
        'color':  'black',
        'weight': 'normal',
        'size': 24
        }

font3 = {'family': 'Opus Special Std',
        'color':  'black',
        'weight': 'normal',
        'size': 25
        }

font4 = {'family': 'Opus Special Std',
        'color':  'black',
        'weight': 'normal',
        'size': 30
        }

font5 = {'family': 'Opus Std',
        'color':  'black',
        'weight': 'normal',
        'size': 18
        }

font6 = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 14
        }

font7 = {'family': 'Opus Special Std',
        'color':  'black',
        'weight': 'normal',
        'size': 12
        }

font8 = {'family': 'Legni FluteV',
        'color':  'black',
        'size': 14
        }

font9 = {'family': 'Opus Special Std',
        'color':  'black',
        'weight': 'normal',
        'size': 14
        }

font10 = {'family': 'Opus Std',
        'color':  'black',
        'weight': 'normal',
        'size': 14
        }

font11 = {'family': 'Opus Special Std',
        'color':  'black',
        'weight': 'normal',
        'size': 20
        }

font12 = {'family': 'Opus Std',
        'color':  'black',
        'weight': 'normal',
        'size': 20
        }

############DATA############
# arrays for data
chars = [ ]
x_moves = [ ]
y_moves = [ ]

#to keep the originals
#read the CSV
with open('alphago.csv', 'rb') as f:
    reader = csv.reader(f)
    raw_data = list(reader)

#OLD--DATA ADDED IN TO TEST
#black_string=";B[qd];B[pq];B[fc];B[ql];B[ld];B[rc];B[re];B[pg];B[ph];B[lf];B[pi];B[kh];B[le];B[kg];B[ne];B[jc];B[jd];B[je];B[if];B[li];B[hf];B[mb];B[ki];B[kk];B[ob];B[lm];B[nb];B[lc];B[ln];B[ll];B[jj];B[hj];B[gj];B[ii];B[ij];B[lo];B[lp];B[lq];B[im];B[fq];B[cn];B[dm];B[gp];B[fr];B[en];B[ep];B[dl];B[kr];B[jb];B[mf];B[nd];B[pj];B[pl];B[ok];B[rl];B[ri];B[pf];B[qh];B[bn];B[bl];B[rg];B[po];B[kq];B[oo];B[on];B[op];B[or];B[oq];B[qr];B[ps];B[rn];B[qn];B[cl];B[ks];B[ol];B[hh];B[dr];B[bq];B[cr];B[dq];B[cp];B[ek];B[bj];B[pb];B[sf];B[ai];B[aj];B[gi];B[fk];B[mc];B[al];B[pm];B[gh]"
#for i in black_string:
#    if i in pos_dict:
#        chars.append(i)

x_moves = chars[0::2]
y_moves = chars[1::2]

#Parameter proximity
moves_off_0 = random.randint(0, 40) #control how close they are
x_pos = [pos_dict[i] for i in x_moves]
y_pos = [pos_dict[i] for i in y_moves]

x_pos = x_pos[moves_off_0:]
y_pos = y_pos[moves_off_0:]
rhy_map  = [ ]
pi_map  = [ ]
yo_map = [ ]

#MAPPING PARAMETERS
def paramMapping(pos, maptype, mini, maxi, rnd):
    for i in range(len(pos)):
        maptype.append(
            round((maxi-mini)*((pos[i] - min(pos)) / (max(pos) - min(pos)))+mini, rnd)
            )

paramMapping(x_pos, rhy_map, 0.15, 1.35, 2) #RHYTHM MAPPING between 0.15 and 1.5
paramMapping(y_pos, pi_map, 40, 53, 0) #PITCH MAPPING between 40 and 53
paramMapping(y_pos, pi_map, 1, 6, 0) #YOWAGIN MAPPING between 1 and 6

#FIXED STUFF
#staff
X_1, Y_1 = np.linspace(0.06, 10.345, 10), np.zeros(10)

#measure lines
X_3, Y_3 = np.linspace(0, 0, 10), np.linspace(4, -38, 10)

#staff  plot
for i in range(len([0,1,2,3,4])):
    ax.plot(X_1, y_off+Y_1+i, linestyle='solid', linewidth=0.5, color='black')

#rhythm = [0.15, 0.15, 0.25, 0.25, 0.4, 0.4, 0.9, 0.9, 1.2, 1.2, 1.2, 1.5, 1.5]
#rhythm = [1]

#pitch
#midi = [40,41,42,43,44,45,46,47,48,49,50,51,52,53]
#yow = ['ge', 'chu', 'chu_uki', 'jo', 'jo_uki', 'kuri']
octaves=[12, 12, 12, 12,  12, 12, 12, 12, 12, 12]


#Accidentals
##acc_all = ['#', 'b', 'n']
acc_all_sp = ['B', u'µ']
acc_all =        [' ', ' ', 'b', 'n',   ' ', ' ', ' ', ' ', 'n', 'b' ]
##acc_all_sp = [u'µ', ' ', 'B', ' ', ' ', ' ', 'B', ' ', ' ', ' ' ]

#Lyrics
hira = [u'ん_________________________','______', u'よ_', '__',
        u'ん__', '______________', '___', u'ん____________', u'ほ__', '__' ]

##hira = [u'[ɕ', u'(ɕ)', u'hoɯ]', u'ぶ', u'て', u'勝', u'負', u'手!']
##hira = [u'勝',u'サ',u'し',u'せ',u'の',u'よ', u'ほ', u'ほ', u'ほ']

#Articulations
art_all = [' ', ' ', ' ']
art_all_sp = [' ', ' ']
stac = '.'
accn = '>'

#noteheads
notes = ['f' ]
#notes = ['X' ]
cols = ['black']

rhy_diff = [ ]
rand_pit = [ ]
gr_pit = [ ]

#Yowagin control
rand_yow = [ ]
rand_art = [ ]
rand_art_sp = [ ]
pitches = [ ]
yowagin = [ ]
yow_sel = [ ]
rand_oct = [ ]
octa = [ ]
rhy_diff = [ ]
accs = [ ]
accs_sp = [ ]
gaccs = [ ]
gaccs_sp = [ ]
noteheads = [ ]
colors = [ ]
lyr = [ ]

#Staff_offset
offset = 1.2 + x_off

#(######CONTROL######)#
#IF we are selecting from later rhythms, we need difference.
ritend = len(rhy_map)
pitend = len(pi_map)

#Rhythm
rhy_diff = rhy_map[moves_off_0:(ritend-(ritend-(moves_off_0+size)))] #selecting all from moves_off on

##create a conditional statement for the other onset
rhy = [offset]
for i in range(len(rhy_diff)):
    try:
            rhy.append(rhy[i]+rhy_diff[i+1])
    except IndexError:
        break

add1 = 2.
add2 = 1.2


#factoring etc.
r_fac = 0.4
rhy = [x*r_fac for x in rhy]
rhy = [x+offset for x in rhy] #we might omit this
size = len(rhy)

#Pitch content
pii = pi_map
pii = pi_map[moves_off_0:(pitend-(pitend-(moves_off_0+size)))]

#Yowagin
rand_yow = yo_map

#Yowagin rules
for i in range(size):
    try:
        if rand_yow[i] is 1:
            rand_yow.append(2)
        if rand_yow[i] is 2:
            yow_sel = random.choice([1, 3])
            rand_yow.append(yow_sel)
        if rand_yow[i] is 3:
            yow_sel = random.choice([2, 4])
            rand_yow.append(yow_sel)
        if rand_yow[i] is 4:
            yow_sel = random.choice([5])
            rand_yow.append(yow_sel)
        if rand_yow[i] is 5:
            yow_sel = random.choice([2, 3, 6])
            rand_yow.append(yow_sel)
        if rand_yow[i] is 6:
            rand_yow.append(5)
    except IndexError:
        None

print('yowagin: ', yo_map[moves_off_0:(pitend-(pitend-(moves_off_0+size)))])

rand_yow = [3, 4, 3, 3,   5, 6, 3, 3, 5, 5]
yowagin = [yow_dict[x] for x in rand_yow]

for i in range(size):
    rand_pit.append(pii[i]) #pitch list
    rand_oct.append(octaves[i])

#Grace note pitch contnent
    gr_pit.append(random.choice(pii))

#Accidentals
    gaccs.append(random.choice(acc_all))
    accs_sp.append(random.choice(acc_all_sp))
    gaccs_sp.append(random.choice(acc_all_sp))

    accs.append(acc_all[i])

#Noteheads
    noteheads.append(random.choice(notes))
    colors.append(random.choice(cols))

#Lyrics
    lyr.append(hira[i])
    rand_art.append(random.choice(art_all))
    rand_art_sp.append(random.choice(art_all_sp))

#Phantom rhythms
phantoms = [1, 2, 2.5, 3, 4, 5, 6, 7, 8]
pitches = [pitch_dict[x] for x in rand_pit]
graces = [pitch_dict[x] for x in gr_pit]

phan = [pha_dict[x] for x in phantoms]
octa = [oct_dict[x] for x in rand_oct]


#Ledger lines
ld = [ ]
lu = [ ]
ladd = [ ]
lsub = [ ]
for i in range(size):
    if rand_pit[i]+rand_oct[i] >= 79 and isinstance(pitches[i], int):
       lu.append('.')
       lsub.append(0.5)
    elif rand_pit[i]+rand_oct[i] >= 79 and isinstance(pitches[i], float):
        lu.append('.')
        lsub.append(0)
    elif rand_pit[i]+rand_oct[i] > 68 and rand_pit[i]+rand_oct[i] < 79 and isinstance(pitches[i], int):
       lu.append('.')
       lsub.append(0)
    elif rand_pit[i]+rand_oct[i] > 68 and rand_pit[i]+rand_oct[i] < 79 and isinstance(pitches[i], float):
        lu.append('.')
        lsub.append(0.5)
    elif rand_pit[i]+rand_oct[i] > 49 and rand_pit[i]+rand_oct[i] < 68:
       lu.append(' ')
       lsub.append(0)
    else:
        lu.append(' ')
        lsub.append(0)

for i in range(size):
    if rand_pit[i]+rand_oct[i] <= 49 and isinstance(pitches[i], int):
       ld.append('.')
       ladd.append(0)
    elif rand_pit[i]+rand_oct[i] <= 49 and isinstance(pitches[i], float):
        ld.append('.')
        ladd.append(0.5)
    else:
        ld.append(' ')
        ladd.append(0)

#Grace ledger lines
gr_ld = [ ]
gr_lu = [ ]
gr_ladd = [ ]
gr_lsub = [ ]
for i in range(size):
    if gr_pit[i]+rand_oct[i] >= 79 and isinstance(graces[i], int):  #got rid of octas...
       gr_lu.append('0')
       gr_lsub.append(0.5)
    elif gr_pit[i]+rand_oct[i] >= 79 and isinstance(graces[i], float):
        gr_lu.append('0')
        gr_lsub.append(0)
    elif gr_pit[i]+rand_oct[i] > 68 and gr_pit[i]+rand_oct[i] < 79 and isinstance(graces[i], int):
       gr_lu.append('0')
       gr_lsub.append(0)
    elif gr_pit[i]+rand_oct[i] > 68 and gr_pit[i]+rand_oct[i] < 79 and isinstance(graces[i], float):
        gr_lu.append('0')
        gr_lsub.append(0.5)
    elif gr_pit[i]+rand_oct[i] > 49 and gr_pit[i]+rand_oct[i] < 68:
       gr_lu.append(' ')
       gr_lsub.append(0)
    else:
        gr_lu.append(' ')
        gr_lsub.append(0)

for i in range(size):
    if gr_pit[i]+rand_oct[i] <= 49 and isinstance(graces[i], int):
       gr_ld.append('0')
       gr_ladd.append(0)
    elif gr_pit[i]+rand_oct[i] <= 49 and isinstance(graces[i], float):
        gr_ld.append('0')
        gr_ladd.append(0.5)
    else:
        gr_ld.append(' ')
        gr_ladd.append(0)


#Treble clef
plt.text(0.1, 1.1+y_off, '&', fontdict=font2)

#Lower beams
#conn = [0] #for auto
conn = [ ] #for manual

#auto-connectors...
#con_1 = random.randint(1, end-5) #number may change
#con_2 = random.randint(con_1+1, end-2)

#for i in range(size):
#    try:
#        con_3 = random.randint(con_2+1, end-1) #con 3???
#        conn.append(con_3)
#    except ValueError:
#        break

#manual
conn.append(range(size))
conn = sum(conn, [ ])
print('conns: ', conn)

ax.plot(np.linspace((rhy[conn[0]])+0.075, (rhy[conn[3]])+0.04, 10), np.linspace(-22.3-y_offset_2, -22.3-y_offset_2, 10), linestyle='solid', linewidth=4, color='black'),
ax.plot(np.linspace((rhy[conn[4]])+0.075, (rhy[conn[9]])+0.04, 10), np.linspace(-22.3-y_offset_2, -22.3-y_offset_2, 10), linestyle='solid', linewidth=4, color='black'),


#Mode: Western or Yowagin
mode = [1,1,0,0, 1,1,1,1,0,0]

#mixed mode
#for i in range(size-len(mode)):
#        mode.append(random.choice([1,1,1,1,1,1,1,1,1,1,0,0,0]))

#Articulations
#for i in range(1):
#    plt.text((rhy[i]+0.12)-0.07, 3.2, stac, fontdict=font2)

plt.text((rhy[3]+0.12)-0.08, 14.3, accn, fontdict=font2)
plt.text((rhy[3]+0.12)-0.07, 13.7, stac, fontdict=font2)
plt.text((rhy[9]+0.12)-0.09, 14.3, accn, fontdict=font2)
plt.text((rhy[9]+0.12)-0.07, 13.7, stac, fontdict=font2)

#Vibrato
#dummy variable for the separated vib
dummy_1 = 1
dummy_2 = 1

#Events
for i in range(size):
    pitch = pitches[i]
    yowa = yowagin[i]
    rhythm = rhy[i]
    acc = accs[i]
    acc_sp = accs_sp[i]
    gacc = gaccs[i]
    gacc_sp = gaccs_sp[i]
    art = rand_art[i]
    art_sp = rand_art_sp[i]
    note_heads = noteheads[i]
    color = colors[i]
    ledger_down = ld[i]
    ledger_up = lu[i]
    ledger_sub = lsub[i]
    ledger_add = ladd[i]
    oct_add = octa[i]
    acc_group = [acc, font2, acc_sp, font1]
    acc_group2 = [acc, font10, acc_sp, font9]
    gacc_group = [gacc, font2, gacc_sp, font1]
    gacc_group2 = [gacc, font10, gacc_sp, font9]
    art_group = [art, font12, art_sp, font11]
    lyrics = lyr[i]
    nota = 0
    stem_len = 8


#lyrics
    plt.text(rhy[i], -7, lyrics,color='Purple', fontdict=font6)

#Accidentals FOR GRACE NOTES
    z2 = random.choice([0,2]) #0 or 2

#Mode choice
    z = random.choice([0,0]) #0 or 2

    if mode[i] is 0: #i for control (also change the other one)
        #add accidentals
        plt.text((rhy[i]+0.026)-0.07, pitches[i]+octa[i], acc_group[z], fontdict=acc_group[z+1]) #nice solution
        #add articulations
        plt.text((rhy[i]+0.12)-0.07, (pitches[i]+octa[i])+1.7, art_group[z], fontdict=art_group[z+1]) #nice solution

#AC
        [
         plt.text(rhythm+0.026, pitch+oct_add, note_heads, fontdict=font3, color=color), #note
         ax.plot(np.linspace(rhythm+0.06, rhythm+0.06, 10), np.linspace((pitch+oct_add)-0.4, -4.9-y_offset_2, 10),
             linestyle='solid', linewidth=0.5, color='black'),  #stem part 1
         ax.plot(np.linspace(rhythm+0.06, rhythm+0.06, 10), np.linspace(-7.7, -22.3-y_offset_2, 10),
             linestyle='solid', linewidth=0.5, color='black'),  #stem part 2
         plt.text(rhythm+0.026, ((pitch+oct_add) - ledger_sub) - 2.95, ledger_up, fontdict=font3), #upper ledger line
         plt.text(rhythm+0.026, ((pitch+oct_add) + ledger_add) + 2.1, ledger_down, fontdict=font3), #lower ledger line
         ]

        #glissandi and parentheses
#        first = 0
#        scnd = first+1
#        plt.plot(np.linspace(rhy[first]+0.06, rhy[scnd]-0.08, 10), np.linspace(pitches[first]+octa[first],pitches[scnd]+octa[scnd], 10),
#                     linestyle='solid', linewidth=0.5, color='black'),  #gliss
#        plt.text(rhy[scnd]-0.083, pitches[scnd]+octa[scnd], '<     >', fontdict=font1) #parens

#Vibrato
#separate vibrato (wes)
        if dummy_1 is 1:
            vib_series = [0.,0.,0.,0.,  0.,0.,0.,1.,  0.,0.,0.7,0.,0.,1.2]
            t = np.linspace(0, vib_series[i], 5000, endpoint=False) #controls length

            car_freq = round(random.uniform(0.1, 5.), 2)
            #am_freq = 0.25 #dal niente
            am_freq = round(random.uniform(0.1, 5.), 2)
            #am_freq = 0.5 #dal niente/al niente
            fm_freq = round(random.uniform(0.1, 5.), 2)
            beta = round(random.uniform(0.1, 1.), 2)
            mul = round(random.uniform(0.1, 0.7), 2)
            mul2 = 1

            a = np.empty(5000)
            b = np.arange(0, 1000, 30)

            ind = np.arange(len(a))
            np.put(a, ind, b)

            car = np.sin(car_freq*(2 * np.pi) * t)
            am_sig = np.sin(am_freq*(2 * np.pi) * t)*mul
            fm_sig = np.sin(fm_freq*(2 * np.pi) * t)
            #pwm = signal.square(2 * np.pi * 30 * t, duty=(sig + 1)/2)
            am_res = am_sig * car
            fm_res = np.cos(car + beta * fm_sig)
            am_fm = am_sig * fm_res

            #new vibrato
            ax.plot(0.08+t+rhy[i], (pitches[i]+octa[i])+am_fm, 'Blue'), #'k'=black

        if rhy_diff[i] > 0.67: #0.5 but make it 0.1 for all to be "vibrated"
            nota = i
            sel_pos = 0

#VIBRATO w/ FM and AM
            t = np.linspace(0, 0, 0, endpoint=False) #zero

            car_freq = round(random.uniform(0.1, 5.), 2)
            #am_freq = 0.25 #dal niente
            am_freq = round(random.uniform(0.1, 5.), 2)
            #am_freq = 0.5 #dal niente/al niente
            fm_freq = round(random.uniform(0.1, 5.), 2)
            beta = round(random.uniform(0.1, 1.), 2)
            mul = round(random.uniform(0.1, 0.7), 2)
            mul2 = 1

            a = np.empty(5000)
            b = np.arange(0, 1000, 30)

            ind = np.arange(len(a))
            np.put(a, ind, b)

            car = np.sin(car_freq*(2 * np.pi) * t)
            am_sig = np.sin(am_freq*(2 * np.pi) * t)*mul
            fm_sig = np.sin(fm_freq*(2 * np.pi) * t)
            am_res = am_sig * car
            fm_res = np.cos(car + beta * fm_sig)
            am_fm = am_sig * fm_res

            #grace note pos
            gr_sel1 = random.choice([-0.43, 0.57])

            #new vibrato
            ax.plot(0.08+t+rhy[nota], (pitches[nota]+octa[nota])+am_fm, 'Blue'), #'k'=black
            if sel_pos is 0:
                None
            elif sel_pos is 1:
                plt.text(rhy[nota]-0.11, (graces[nota])+octa[nota], notes[0], fontdict=font7, color=cols[0]), #grace note1
                plt.text((rhythm-0.11+0.026)-0.07, (graces[nota])+octa[i], gacc_group[z2], fontdict=gacc_group2[z2+1]) #accidentals
                plt.plot(np.linspace(rhy[nota]-0.1, rhy[nota]-0.1, 10), np.linspace((graces[nota]+octa[nota]), stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem1
                plt.plot(np.linspace(rhy[nota]-0.09, rhy[nota]-0.07, 10), np.linspace(stem_len+y_off, stem_len+y_off, 10), linestyle='solid', linewidth=2, color='black'), #single connector
            elif sel_pos is 2:
                plt.text(rhy[nota]-0.11, graces[nota]+octa[nota], notes[0], fontdict=font7, color=cols[0]), #grace note1
                plt.text(rhy[nota]-0.17, graces[nota]+octa[nota], notes[0], fontdict=font7, color=cols[0]), #grace note2
                plt.plot(np.linspace(rhy[nota]-0.1, rhy[nota]-0.1, 10), np.linspace((graces[nota]+octa[nota]), stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem1
                plt.plot(np.linspace(rhy[nota]-0.16, rhy[nota]-0.16, 10), np.linspace((graces[nota]+octa[nota]), stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem2
                plt.plot(np.linspace(rhy[nota]-0.15, rhy[nota]-0.11, 10), np.linspace(stem_len+y_off, stem_len+y_off, 10), linestyle='solid', linewidth=2, color='black'), #connector for 2
                plt.text((rhythm-0.17+0.026)-0.07, (graces[i])+octa[1], acc_group[z2], fontdict=acc_group2[z2+1]) #accidental for 2
                plt.text(rhythm-0.11, (graces[nota-1]+octa[1] - grsub) - 3.46, grlu, fontdict=font3), #upper ledger line for grace1
                plt.text(rhythm-0.17, (graces[nota]+octa[1] - grsub) - 3.46, grlu, fontdict=font3), #upper ledger line for grace2
            elif sel_pos is 3:
                plt.text(rhy[nota]-0.11, pitches[nota]+octa[nota], notes[0], fontdict=font7, color=cols[0]), #grace note1
                plt.text(rhy[nota]-0.17, pitches[nota]+octa[nota], notes[0], fontdict=font7, color=cols[0]), #grace note2
                plt.text(rhy[nota]-0.23, pitches[nota]+octa[nota], notes[0], fontdict=font7, color=cols[0]), #grace note3
                plt.plot(np.linspace(rhy[nota]-0.1, rhy[nota]-0.1, 10), np.linspace((pitches[nota]+octa[nota]), stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem1
                plt.plot(np.linspace(rhy[nota]-0.16, rhy[nota]-0.16, 10), np.linspace((pitches[nota]+octa[nota]), stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem2
                plt.plot(np.linspace(rhy[nota]-0.21, rhy[nota]-0.21, 10), np.linspace((pitches[nota]+octa[nota]), stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem3
                plt.plot(np.linspace(rhy[nota]-0.2, rhy[nota]-0.11, 10), np.linspace(stem_len+y_off, stem_len+y_off, 10), linestyle='solid', linewidth=2, color='black'), #connector for 3
            elif sel_pos is 4:
                plt.text(rhy[nota]-0.11, pitches[nota]+octa[nota], notes[0], fontdict=font7, color=cols[0]), #grace note1
                plt.text(rhy[nota]-0.17, pitches[nota]+octa[nota], notes[0], fontdict=font7, color=cols[0]), #grace note2
                plt.text(rhy[nota]-0.23, pitches[nota]+octa[nota], notes[0], fontdict=font7, color=cols[0]), #grace note3
                plt.text(rhy[nota]-0.29, pitches[nota]+octa[nota], notes[0], fontdict=font7, color=cols[0]), #grace note4
                plt.plot(np.linspace(rhy[nota]-0.1, rhy[nota]-0.1, 10), np.linspace((pitches[nota]+octa[nota]), stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem1
                plt.plot(np.linspace(rhy[nota]-0.16, rhy[nota]-0.16, 10), np.linspace((pitches[nota]+octa[nota]), stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem2
                plt.plot(np.linspace(rhy[nota]-0.21, rhy[nota]-0.21, 10), np.linspace((pitches[nota]+octa[nota]), stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem3
                plt.plot(np.linspace(rhy[nota]-0.27, rhy[nota]-0.27, 10), np.linspace((pitches[nota]+octa[nota]), stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem4
                plt.plot(np.linspace(rhy[nota]-0.26, rhy[nota]-0.11, 10), np.linspace(stem_len+y_off, stem_len+y_off, 10), linestyle='solid', linewidth=2, color='black'), #connector for 4

    #MC_4
    if mode[i] is 1:
        [
        plt.text(rhythm+0.026, yowa, note_heads, fontdict=font4, color=color), #note
        ax.plot(np.linspace(rhythm+0.06, rhythm+0.06, 10), np.linspace(yowa-0.4, -4.9-y_offset_2, 10),
            linestyle='solid', linewidth=0.5, color='black'),  #stem part 1
        ax.plot(np.linspace(rhythm+0.06, rhythm+0.06, 10), np.linspace(-7.7, -22.3-y_offset_2, 10),
            linestyle='solid', linewidth=0.5, color='black'),  #stem part 2
        ]
        #add articulations
        plt.text((rhy[i]+0.12)-0.07, yowa+1.5, art_group[z], fontdict=art_group[z+1]) #nice solution

        #glissandi and parentheses
#        first = 4
#        scnd = first+1
#        plt.plot(np.linspace(rhy[first]+0.06, rhy[scnd]-0.08, 10), np.linspace(yowagin[first],yowagin[scnd], 10),
#            linestyle='solid', linewidth=0.5, color='black'),  #gliss
#        plt.text(rhy[scnd]-0.04, yowagin[scnd], '<    >', fontdict=font1) #parens
#       kuri (indiv)

#YC_issue --USE INTEGER FOR RANDOM, FLOAT FOR CONTROL
        if rand_yow[i] is 6:
            ax.add_line(Line2D([rhy[i]-0.02, rhy[i]+0.14],  [-0.1+5.0, -0.1+5.0], #
                  linestyle='-', linewidth=1, color='Purple'))
            ax.add_line(Line2D([rhy[i]-0.02, rhy[i]+0.14],  [-0.1+3.75, -0.1+3.75], #1.25
                  linestyle='-', linewidth=1, color='Purple'))
        #jo-uki (indiv)
        if rand_yow[i] is 5:
            ax.add_line(Line2D([rhy[i]-0.02, rhy[i]+0.14],  [-0.1+3.75, -0.1+3.75], #1.25
                  linestyle='-', linewidth=1, color='Purple'))
        #chu-uki (indiv)
        if rand_yow[i] is 3:
            ax.add_line(Line2D([rhy[i]-0.02, rhy[i]+0.14], [-0.1+1.25, -0.1+1.25], #1.25
                  linestyle='-', linewidth=1, color='Purple'))

#separate vibrato (yow)
        if dummy_2 is 1:
            vib_series = [1.,0.8,0.,0.,  0.,0.5,0.,1.2,  0.,0.,0.7,1.,0.,1.2]
            t = np.linspace(0, vib_series[i], 5000, endpoint=False) #controls length
            car_freq = round(random.uniform(0.1, 5.), 2)
            #am_freq = 0.25 #dal niente
            am_freq = round(random.uniform(0.1, 5.), 2)
            #am_freq = 0.5 #dal niente/al niente
            fm_freq = round(random.uniform(0.1, 5.), 2)
            beta = round(random.uniform(0.1, 1.), 2)
            mul = round(random.uniform(0.1, 0.7), 2)
            mul2 = 1

            a = np.empty(5000)
            b = np.arange(0, 1000, 30)

            ind = np.arange(len(a))
            np.put(a, ind, b)

            car = np.sin(car_freq*(2 * np.pi) * t)
            am_sig = np.sin(am_freq*(2 * np.pi) * t)*mul
            fm_sig = np.sin(fm_freq*(2 * np.pi) * t)
            am_res = am_sig * car
            fm_res = np.cos(car + beta * fm_sig)
            am_fm = am_sig * fm_res

            #new vibrato
            ax.plot(0.08+t+rhy[i], yowagin[i]+am_fm, 'Blue'), #'k'=black

        if rhy_diff[i] > 0.77: #0.5 but make it 0.1 for all to be "vibrated" #yow
            nota = i
            sel_pos = random.randint(0, 1)

            #grace note pos
            gr_sel2 = random.choice([-0.73, 0.97])

           #VIBRATO w/ FM and AM
            t = np.linspace(0, 0, 0, endpoint=False) #zero
            car_freq = round(random.uniform(0.1, 5.), 2)
            #am_freq = 0.25 #dal niente
            am_freq = round(random.uniform(0.1, 5.), 2)
            #am_freq = 0.5 #dal niente/al niente
            fm_freq = round(random.uniform(0.1, 5.), 2)
            beta = round(random.uniform(0.1, 1.), 2)
            mul = round(random.uniform(0.1, 0.7), 2)

            a = np.empty(5000)
            b = np.arange(0, 1000, 30)

            ind = np.arange(len(a))
            np.put(a, ind, b)

            car = np.sin(car_freq*(2 * np.pi) * t)
            am_sig = np.sin(am_freq*(2 * np.pi) * t)*mul
            fm_sig = np.sin(fm_freq*(2 * np.pi) * t)
            am_res = am_sig * car
            fm_res = np.cos(car + beta * fm_sig)
            am_fm = am_sig * fm_res
           #new vibrato
            ax.plot(0.08+t+rhy[nota], yowagin[i]+am_fm, 'Blue'), #'k'=black

            if sel_pos is 0:
                None
            elif sel_pos is 1:
                plt.text(rhy[nota]-0.08, yowagin[nota]+gr_sel2, notes[0], fontdict=font7, color=cols[0]), #grace note1
                plt.plot(np.linspace(rhy[nota]-0.07, rhy[nota]-0.07, 10), np.linspace(yowagin[nota]+gr_sel2, stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem1
                plt.plot(np.linspace(rhy[nota]-0.06, rhy[nota]-0.02, 10), np.linspace(stem_len+y_off, stem_len+y_off, 10), linestyle='solid', linewidth=2, color='black'), #single connector
            elif sel_pos is 2:
                plt.text(rhy[nota]-0.11, yowagin[nota], notes[0], fontdict=font7, color=cols[0]), #grace note1
                plt.text(rhy[nota]-0.17, yowagin[nota], notes[0], fontdict=font7, color=cols[0]), #grace note2
                plt.plot(np.linspace(rhy[nota]-0.1, rhy[nota]-0.1, 10), np.linspace(yowagin[nota], stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem1
                plt.plot(np.linspace(rhy[nota]-0.16, rhy[nota]-0.16, 10), np.linspace(yowagin[nota], stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem2
                plt.plot(np.linspace(rhy[nota]-0.15, rhy[nota]-0.11, 10), np.linspace(stem_len+y_off, stem_len+y_off, 10), linestyle='solid', linewidth=2, color='black'), #connector for 2
            elif sel_pos is 3:
                plt.text(rhy[nota]-0.11, yowagin[nota], notes[0], fontdict=font7, color=cols[0]), #grace note1
                plt.text(rhy[nota]-0.17, yowagin[nota], notes[0], fontdict=font7, color=cols[0]), #grace note2
                plt.text(rhy[nota]-0.23, yowagin[nota], notes[0], fontdict=font7, color=cols[0]), #grace note3
                plt.plot(np.linspace(rhy[nota]-0.1, rhy[nota]-0.1, 10), np.linspace(yowagin[nota], stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem1
                plt.plot(np.linspace(rhy[nota]-0.16, rhy[nota]-0.16, 10), np.linspace(yowagin[nota], stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem2
                plt.plot(np.linspace(rhy[nota]-0.21, rhy[nota]-0.21, 10), np.linspace(yowagin[nota], stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem3
                plt.plot(np.linspace(rhy[nota]-0.2, rhy[nota]-0.11, 10), np.linspace(stem_len+y_off, stem_len+y_off, 10), linestyle='solid', linewidth=2, color='black'), #connector for 3
            elif sel_pos is 4:
                plt.text(rhy[nota]-0.11, yowagin[nota], notes[0], fontdict=font7, color=cols[0]), #grace note1
                plt.text(rhy[nota]-0.17, yowagin[nota], notes[0], fontdict=font7, color=cols[0]), #grace note2
                plt.text(rhy[nota]-0.23, yowagin[nota], notes[0], fontdict=font7, color=cols[0]), #grace note3
                plt.text(rhy[nota]-0.29, yowagin[nota], notes[0], fontdict=font7, color=cols[0]), #grace note4
                plt.plot(np.linspace(rhy[nota]-0.1, rhy[nota]-0.1, 10), np.linspace(yowagin[nota], stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem1
                plt.plot(np.linspace(rhy[nota]-0.16, rhy[nota]-0.16, 10), np.linspace(yowagin[nota], stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem2
                plt.plot(np.linspace(rhy[nota]-0.21, rhy[nota]-0.21, 10), np.linspace(yowagin[nota], stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem3
                plt.plot(np.linspace(rhy[nota]-0.27, rhy[nota]-0.27, 10), np.linspace(yowagin[nota], stem_len+y_off, 10), linestyle='solid', linewidth=0.5, color='black'),  #stem4
                plt.plot(np.linspace(rhy[nota]-0.26, rhy[nota]-0.11, 10), np.linspace(stem_len+y_off, stem_len+y_off, 10), linestyle='solid', linewidth=2, color='black'), #connector for 4

for i in range(2):
    ax.plot(np.linspace(phan[i]+0.06, phan[i]+0.06, 10), np.linspace(4+y_off, 15+y_off, 10), linestyle='solid', linewidth=0.5, color='gray'),  #stems
for i in range(1):
    ax.plot(np.linspace(phan[i+2]+0.06, phan[i+2]+0.06, 10), np.linspace(-3.5+y_off, 15+y_off, 10), linestyle='solid', linewidth=0.5, color='gray'),  #stems
for i in range(2):
    ax.plot(np.linspace(phan[i+3]+0.06, phan[i+3]+0.06, 10), np.linspace(4+y_off, 15+y_off, 10), linestyle='solid', linewidth=0.5, color='gray'),  #stems
for i in range(1):
    ax.plot(np.linspace(phan[i+5]+0.06, phan[i+5]+0.06, 10), np.linspace(4+y_off, 15+y_off, 10), linestyle='solid', linewidth=0.5, color='gray'),  #stems
for i in range(1):
    ax.plot(np.linspace(phan[i+6]+0.06, phan[i+6]+0.06, 10), np.linspace(-3.5+y_off, 15+y_off, 10), linestyle='solid', linewidth=0.5, color='gray'),  #stems
for i in range(2):
    ax.plot(np.linspace(phan[i+7]+0.06, phan[i+7]+0.06, 10), np.linspace(4+y_off, 15+y_off, 10), linestyle='solid', linewidth=0.5, color='gray'),  #stems


for i in range(len(phan)):
    try:
        [
        ax.plot(np.linspace((phan[0])+0.075, (phan[4])+0.045, 10), np.linspace(15+y_off, 15+y_off, 10), linestyle='solid', linewidth=4, color='gray'), #1
        ax.plot(np.linspace((phan[1])+0.075, (phan[2])+0.045, 10), np.linspace(14+y_off, 14+y_off, 10), linestyle='solid', linewidth=4, color='gray'), #1
        ax.plot(np.linspace((phan[5])+0.075, (phan[8])+0.04, 10), np.linspace(15+y_off, 15+y_off, 10), linestyle='solid', linewidth=4, color='gray'), #2
        ]
    except IndexError:
        break

plt.yticks([]),
plt.xticks([])

###INDEX

####pitch space####
pit = plt.axes([0.03, 0.43, 0.96, 0.25])#axes([left, bottom, width, height])
ax.add_artist(pit) #did the trick

pit.add_line(Line2D([0, 1], [0.6, 0.6], transform=pit.transAxes,
                  linestyle='-', linewidth=1, color='Black'))
pit.add_line(Line2D([0, 1], [0.45, 0.45], transform=pit.transAxes,
                  linestyle='-', linewidth=1, color='Black'))
pit.add_line(Line2D([0, 1], [0.3, 0.3], transform=pit.transAxes,
                  linestyle='-', linewidth=1, color='Black'))

x0, x1, y0, y1 = pit.axis()
pit.axis((x0*9,
          x1*9,
          y0*2-1,
          y1*2-1))

pit.set_zorder(-1)
pit.spines['top'].set_visible(False)
pit.spines['bottom'].set_visible(False)
pit.spines['left'].set_visible(False)
pit.spines['right'].set_visible(False)
plt.yticks([]),
plt.xticks([])


####parameters####
# this is an inset axes over the main axes
par = plt.axes([0.03, 0.22, 0.96, 0.2])#axes([left, bottom, width, height])
ax.add_artist(par) #did the trick

#create the param lines
par.add_line(Line2D([0, 1], [1.0, 1.0], transform=par.transAxes,
                  linestyle='-', linewidth=3.2, color='Red'))
par.add_line(Line2D([0, 1], [0.75, 0.75], transform=par.transAxes,
                  linestyle=':', linewidth=1, color='Black'))
par.add_line(Line2D([0, 1], [0.5, 0.5], transform=par.transAxes,
                  linestyle='--', linewidth=1, color='Black'))
par.add_line(Line2D([0, 1], [0.25, 0.25], transform=par.transAxes,
                  linestyle=':', linewidth=1, color='Black'))
par.add_line(Line2D([0, 1], [0.0, 0.0], transform=par.transAxes,
                  linestyle='-', linewidth=1.2, color='Red'))
x0, x1, y0, y1 = par.axis()
par.axis((x0*9,
          x1*9,
          y0*2-1,
          y1*2-1))

moves_off_1 = random.randint(0, 40)
moves_off_2 = random.randint(0, 40)

x_pos_1 = x_pos[moves_off_1:]
x_pos_2 = x_pos[moves_off_2:]
y_pos_1 = y_pos[moves_off_1:]
y_pos_2 = y_pos[moves_off_2:]

x_1  = [0]
y_1  = [0]
x_2  = [0]
y_2  = [0]

#Playing parameters
#differences
for i in range(len(x_pos_1)):
    try:
        x_1.append(x_1[i]+x_pos_1[i+1])
    except IndexError:
        break
factor_1 = 17/max(x_1) #smaller dividend => compression on the x axis.
x_1 = np.multiply(x_1, factor_1)

for i in range(len(x_pos_2)):
    try:
        x_2.append(x_2[i]+x_pos_2[i+1])
    except IndexError:
        break
factor_2 = 12/max(x_2)
x_2 = np.multiply(x_2, factor_2)

#normalize between 1 and -1
for i in range(len(y_pos_1)):
    y_1.append(
        round(2*((y_pos_1[i] - min(y_pos_1)) / (max(y_pos_1) - min(y_pos_1)))-1, 3)
        )
for i in range(len(y_pos_2)):
    y_2.append(
        round(2*((y_pos_2[i] - min(y_pos_2)) / (max(y_pos_2) - min(y_pos_2)))-1, 3)
        )

param_x_1 = np.array(x_1)
param_y_1 = np.array(y_1)
param_x_2 = np.array(x_2)
param_y_2 = np.array(y_2)

#divi = 6.65 #more = higher amp
divi = 4.65
low_lim = 17.2
factor_y = divi/max(y_1) #dividend = upper limit
param_y_1 = np.multiply(param_y_1, factor_y)-low_lim #-15.2 -->lower limit
factor_y_2 = divi/max(y_2) #dividend = upper limit
param_y_2 = np.multiply(param_y_2, factor_y_2)-low_lim


#1
#0.01 cozunurlukte random sayilar

#singles   TO ADD PARAMETERS TO THE SINGLE LINES
f_1 = interpolate.PchipInterpolator(param_x_1, param_y_1, 0)
f_2 = interpolate.PchipInterpolator(param_x_2, param_y_2, 0)

#red horizontal
xnew1_1 = np.arange((rhy[conn[0]]+0.075), (rhy[conn[3]]+0.04), 0.001)
xnew1_2 = np.arange((rhy[conn[4]]+0.075), (rhy[conn[9]]+0.04), 0.001)

#assigning y funcs
ynew1_1 = f_1(xnew1_1)
ynew1_2 = f_1(xnew1_2)

xnew2_1 = np.arange((rhy[conn[0]]+0.075), (rhy[conn[0]]+0.04), 0.001)

ynew2_1 = f_2(xnew2_1)

ax.plot(xnew1_1, ynew1_1, '-', color='Red', alpha = 1)
ax.plot(xnew1_2, ynew1_2, '-', color='Red')

par.set_zorder(-1)
par.spines['top'].set_visible(False)
par.spines['bottom'].set_visible(False)
par.spines['left'].set_visible(False)
par.spines['right'].set_visible(False)

plt.yticks([])
plt.xticks([])

#global y-limits
ax.set_ylim(-35, 29)

fig.set_tight_layout(True)
plt.yticks([])
plt.xticks([])

plt.show()

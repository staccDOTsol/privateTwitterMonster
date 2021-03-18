DAYS_PER_EPISODE = 3 * 31
MINUTES_PER_EPISODE = 24*DAYS_PER_EPISODE#*60
A=1#print(MINUTES_PER_EPISODE)
NUM_EPISODES = 1
signals = {"futs": {'BTC': 'a475710c-41d7-4aa0-8082-53166dc59034',
'ETH': '59fd8f40-6415-47fd-a6ed-f3f6b9c2aef0',
'XRP': '64e93539-4d35-4f6c-b9de-fd521b6da465',
'BNB': 'bfdf8c5c-12dd-430a-9589-a1966bd3e947',
'ADA': 'aae073fb-1e93-4055-b979-2484c5b9c673',
'EOS': '2a4e1c21-106f-4f49-8aa3-be1cdd079858',
'BCH': 'd126bb75-c9ba-4d6c-834b-9b2d2f43b8a1'}, 'spot':{
'ETH': '4edd51ec-57bb-4e3b-b13c-4a84fba623ca', 
'XRP': '73fafffe-16f9-435c-bd46-3d69e03722b3',
'BNB': '9bab001f-b2cc-4d3f-9c18-6157cba01f9c',
'ADA': '7e839951-0460-4dfb-80ef-3ff4e503e80d',
'EOS': '040ee5dd-625e-4648-ad3c-c5ca6b3a9f14',
'BCH': '797ca7f0-2255-4a6a-88cc-ce1725d3f422'}}
import ccxt
import gspread
from oauth2client.service_account import ServiceAccountCredentials

apikeys = {
"tier": "team",
"nick": "jare", 
"type": "spot", 
"key": "31YHFx31DnLBvNfnohFxhLAnHZtsGrJnkbpbeswutJaetzdflS753Uqnj6pCUB61", 
"secret": "3b9VCQIJlXDGOKEux5odKJMDQhKogsv16KQvd46sk4lTkAaszfftMEwIFraVpY31"}


exchanges = {}
#for apis in apikeys:
	#if apis['type'] == 'us':

exchanges[apikeys['key']] = (ccxt.binance({
	'apiKey': apikeys['key'],
	'secret': apikeys['secret'],
	'timeout': 30000,
	'enableRateLimit': True,
	'options':{ 'defaultType': 'future'}
}))
binspots = (ccxt.binance({
	'apiKey': apikeys['key'],
	'secret': apikeys['secret'],
	'timeout': 30000,
	'enableRateLimit': True
}))
balances = {}
for exchange in exchanges:
	balances[exchange] = {}
	"""
	try:
		bal = exchanges[exchange].fetchBalance()
		for b in bal:
			if b == 'total':
				for coin in bal[b]:
					
					balances[exchange][coin] = bal[b][coin]
	except Exception as e:
		print(e)
	"""
		#exchanges.remove(exchange)
"""
price = exchanges["31YHFx31DnLBvNfnohFxhLAnHZtsGrJnkbpbeswutJaetzdflS753Uqnj6pCUB61"].fetchTicker('BTC' + "/USDT")['info']['lastPrice']	
													
o = exchanges[exchange].createOrder ('BTC' + '/USDT', 'market', 'buy', 11 / price, None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
print(o)
#o = exchanges[exchange].createOrder ('BTC' + '/USDT', 'market', 'sell', balances[exchange]['BTC'], None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
#print(o)

exchange = ccxt.binance({
	'apiKey': '31YHFx31DnLBvNfnohFxhLAnHZtsGrJnkbpbeswutJaetzdflS753Uqnj6pCUB61',
	'secret': '3b9VCQIJlXDGOKEux5odKJMDQhKogsv16KQvd46sk4lTkAaszfftMEwIFraVpY31',
	'timeout': 30000,
	'enableRateLimit': True,
})

"""
import os

symbols = []
import pandas as pd
dontdo = ['DEFI', 'BTC', 'ETH']
import asciichart
import requests

r = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr").json()
symbols = []
for sym in r:
	if 'BTC' in sym['symbol']:
		if sym['symbol'].replace('BTC','') not in symbols:
			if sym['symbol'].replace('BTC','') not in dontdo:
				symbols.append(sym['symbol'].replace('BTC',''))
import random, string
def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

print(randomword(20))
balances = {}
for exchange in exchanges:
	balances[exchange] = {}
	"""
	try:
		bal = exchanges[exchange].fetchBalance()
		for b in bal:
			if b == 'total':
				for coin in bal[b]:
					
					balances[exchange][coin] = bal[b][coin]
					 
					if coin != 'USDT' and balances[exchange][coin] > 0:
						try:
							o = exchanges[exchange].createOrder (coin + '/USDT', 'market', 'sell', balances[exchange][coin], None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
							print(o)
						except:
							abc=123
					
	except:
		abc=123			
	"""	
"""
try:
	#o = exchanges[exchange].createOrder ('BTC' + '/USDT', 'market', 'buy', balances['USDT'] / price, None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
	#print(o)
	o = exchanges[exchange].createOrder ('ETH' + '/USDT', 'market', 'sell', balances['ETH'], None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
	print(o)
except Exception as e:
	print(e)
"""
os.environ["sentiment"] = str(0.2)
os.environ["stop"] = str(0.1)
os.environ["learnwin"] = str(20)
os.environ["learnlose"] = str(13)
os.environ["lorm"] = 'more'
os.environ["balancedivisor"] = str(14)
import pylab as plt2
TRADING_FEE_MULTIPLIER = .99925 #this is the trading fee on binance VIP level 0 if using BNB to pay fees
import numpy as np #pip install numpy
from tqdm import tqdm #pip install tqdm
from binance.client import Client #pip install python-binance
import pandas as pd #pip install pandas
from datetime import datetime
from datetime import timedelta
import sys, linecache
def PrintException():

	exc_type, exc_obj, tb = sys.exc_info()
	f = tb.tb_frame
	lineno = tb.tb_lineno
	filename = f.f_code.co_filename
	linecache.checkcache(filename)
	line = linecache.getline(filename, lineno, f.f_globals)
	string = 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj)
	print	(string)
	sleep(0.1)
import random
"""
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
analyser = SentimentIntensityAnalyzer() 
new_words = {
'bullish': 0.75, 
'bearish': -0.75,
'neutral': 0,
}
analyser.lexicon.update(new_words) 
def sentiment_analyzer_scores(sentence):
	score = analyser.polarity_scores(sentence)
	return(score)
"""
users = ['ArminVanBitcoin','aantonop', 'elonmusk','officialmcafee','vitalikbuterin','satoshilite','pmarca','rogerkver','aantonop', 'ErikVoorhees','nickszabo4','CryptoYoda1338','bgarlinghouse','lopp','barrysilbert','ToneVays','vinnylingham','APompliano','CharlieShrem','gavinandresen','CryptoCobain','winklevoss','MaheshSashital','jimmysong','simoncocking','CryptoHustle','dtapscott','JoelKatz','TimDraper','cryptoSqueeze','laurashin','TheCryptoDog','balajis','CremeDeLaCrypto','iamjosephyoung','Crypto_Bitlord','giacomozucco','woonomic','parabolictrav','el33th4xor','Melt_Dem','haydentiff','CryptoDonAlt','Fisher85M','jonmatonis','stephantual','Beastlyorion','ummjackson','brucefenton','ProfFaustus','dahongfei','kyletorpey','TuurDemeester','TheBlueMatt','slushcz','pierre_rochard','francispouliot_','AriannaSimpson','LukeDashjr','justmoon','nathanielpopper','bytemaster7','prestonjbyrne','saifedean','TheCryptoMonk','muneeb','AaronvanW','diiorioanthony','_jonasschnelli_','alansilbert','BitcoinByte','alexsunnarborg','disruptepreneur','chrislarsensf','bitstein','valkenburgh','JedMcCaleb','avsa','nbougalis','adamludwin','oleganza','_jillruth','bendavenport','JackMallers','Xentagz','CryptoTrooper_','ofnumbers','alexbosworth','SDLerner','matthewroszak','CaitlinLong_','TokenHash','Dan_Jeffries1','AlyseKilleen','mikebelshe','DanielKrawisz','conniegallippi','Snyke','minhokim','jamieCrypto','LarryBitcoin','SHodyEsq']

import requests


import tweepy

# Authenticate to Twitter
auth = tweepy.OAuthHandler("k7cAVcyoOGsedbSRl5UtoIN6d", "OcbyrBhAYGEWLRanR2U71dLy9RggyXEZIRg1PJ1GDa6wULX0ra")
auth.set_access_token("4352022141-X1y3ZFJ22D8mBe5ELvbOij5OtqVxRVvOlMULwFu", "IU6j4yytn3MVZRwn1F3ChygVPICQ3OHQTa7mMHznZniKU")

# Create API object
api = tweepy.API(auth)
donetweets = []

SMA_LOW = 40
SMA_HIGH = 150

def compute_sma(data, window):
	sma = data.rolling(window=window).mean()
	return sma

import requests

r = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr").json()
symbols = []
for sym in r:
	if 'USDT' in sym['symbol']:
		if sym['symbol'].replace('USDT','') not in symbols:
			if sym['symbol'].replace('USDT','') not in dontdo:
				symbols.append(sym['symbol'].replace('USDT',''))
#select cryptocurrencies you'd like to gather and time interval
ratios = symbols
"""
from datetime import datetime
from datetime import timedelta

from tqdm import tqdm
merge = False
import pandas as pd
import numpy as np
import os.path
from os import path
import requests
"""
"""
#select cryptocurrencies you'd like to gather and time interval
ratios = symbols
for sym in ratios:
	try:
		if path.exists(sym):
		
			temp_df = pd.read_csv(sym, usecols = ['time',sym+'-USD_close'])
			if merge == False:
				df = temp_df
				A=1#print(df)
			else:
				df2 = pd.merge(df,temp_df,how='left',on='time')
				#df2.fillna('unknown', inplace=True)

				if(len(df2) > 1):
					df = df2
				else:   
					
					A=1#print(df2)
			merge = True
	except Exception as e:
		A=1#print(e)
A=1#print(df)
for col in df.columns:
	if col != 'time':
		df[col] = df[col].astype(np.float64)

for i in tqdm(range(len(df))):
	try:
		df['time'][i] = datetime.fromtimestamp(int(df['time'][i]/1000))
	except Exception as e:
		A=1#print(e)
		abc=123
df.to_csv('from_binance_hourly')
"""
cs = ratios
thetime = datetime.utcnow()- timedelta(hours =1.0)


threeyears = thetime#datetime.utcnow() - timedelta(days=DAYS_PER_EPISODE)

START_TIME = '20 Dec, 2020'#26 Jan
END_TIME = '20 Feb, 2021'
api_key = "EFnR9fOpJhrNYUYEJZkWu2iLeTFhYZH1fp4aIZgEIl19D9bN1WrsU9vbfIcO0GME"
api_secret = "Jm9G1H9Y2QosonQBaCr67A3LN8Zz88DqWeQOc2Lm6OJOHlDuhRfpQrSdVlYWtMt7"
 

client = Client(api_key=api_key,api_secret=api_secret)
import time
merge = False
class Memory:
	def __init__(self): 
		self.clear()

	def clear(self): 
		self.actions = []
		
	def add_to_memory(self, new_action): 
		self.actions.append(new_action)
ss = []
tweets2 = {}
import pandas as pd
import numpy as np
import os
#os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
#os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
import json
"""
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

spark = sparknlp.start()
MODEL_NAME='classifierdl_use_sarcasm'
documentAssembler = DocumentAssembler()\
	.setInputCol("text")\
	.setOutputCol("document")
	
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")


sentimentdl = ClassifierDLModel.pretrained(name=MODEL_NAME)\
	.setInputCols(["sentence_embeddings"])\
	.setOutputCol("sentiment")

nlpPipeline = Pipeline(
	  stages = [
		  documentAssembler,
		  use,
		  sentimentdl
	  ])
## Generating Example Files ##

empty_df = spark.createDataFrame([['']]).toDF("text")
"""
import smtplib
import json
ss = []
dos = ['DOGE', 'ETH', 'BTC', 'MARS', 'BCH', 'BNB', 'ADA', 'XRP', 'LTC', 'XLM', 'XEM', 'AAVE', 'EOS', 'BSV', 'XMR', 'TRX', 'FTT', 'XTZ', 'SNX', 'MKR', 'FIL', 'XSM']
with open (os.environ['lorm'], "r") as f:
	ss = json.loads(f.read())
#print(ss)
high = 88888888888888888888888888888
for u in ss:
	for s in ss[u]:
		if s['dt']['$date'] < high:
			high = s['dt']['$date']
A=1#print(high)
from time import sleep
from bson import json_util
#json.loads(aJsonString, object_hook=json_util.object_hook)
import threading	
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.dates import DateFormatter

class Env:
	def __init__(self, ratios, df):
		self.ratios = ratios
		self.weights = {}
		self.stopsizes = {}
		self.signalIds = {}
		self.signalIds2 = {}
		with open ("sls3.json", 'r') as slsjson:

			self.stoplosses = json.loads(slsjson.read())
		with open ("stops3.json", 'r') as slsjson:

			self.stopsizes = json.loads(slsjson.read())
		with open ("bals3.json", 'r') as slsjson:

			self.thisbal = json.loads(slsjson.read())
		with open ("weights3.json", 'r') as slsjson:

			self.weights = json.loads(slsjson.read())
		print(self.weights)
		maxi = 0
		for u in self.weights:
			if self.weights[u] > maxi:
				maxi = self.weights[u]
		print(maxi)
		with open ("ubs3.json", 'r') as slsjson:

			self.user_buys = json.loads(slsjson.read())
		with open ("bs3.json", 'r') as slsjson:

			self.buys = json.loads(slsjson.read())
		with open ("sigids3.json", 'r') as slsjson:

			self.signalIds = json.loads(slsjson.read())
		with open ("sigids32.json", 'r') as slsjson:

			self.signalIds2 = json.loads(slsjson.read())
			print(self.signalIds2)
		"""		
		urll= "https://zignaly.com/api/signals.php?key=ddf42e3aad15cb4ee14af970b8c7e812&pair=BTCUSDT&type=exit&exchange=binance&exchangeAccountType=futures&signalId=" + self.signalIds['31YHFx31DnLBvNfnohFxhLAnHZtsGrJnkbpbeswutJaetzdflS753Uqnj6pCUB61']['BTC'][0]
				
		print(urll)
		result = requests.get(urll)
		print(result)
		"""
		for exchange in exchanges:
			if exchange not in self.stoplosses:
				self.stoplosses[exchange] = {}
				self.stopsizes[exchange] = {}
				self.signalIds[exchange] = {}
				self.signalIds2[exchange] = {}
			if exchange not in self.thisbal:
				self.thisbal[exchange] = {}
			self.weights[exchange] = {}
			for u in users:
				self.weights[exchange][u] = 1
			#self.stopsizes[exchange] = {}
		#print(self.stopsizes)
		
		self.returns = pd.DataFrame({},columns=['time', 'equity'])
		A=1#print(df)
		with open ("ids3.json", 'r') as idsjson:

			self.donetweets = json.loads(idsjson.read())
		#self.donetweets = []
		self.balance = 1000
		self.iloc2 = 0
		self.net_worth = None
		self.rcount = 0
		self.buysigs = []
		self.sellsigs = []
		self.main_df = df
		self.reset()
			
	def reset(self):
		self.balances = {}
		self.thishold = {}

		for ratio in self.ratios:
			self.balances[ratio] = 0.0
			self.thishold[ratio] = []
		A=1#print(len(self.main_df))
		self.iloc = 0#random.randint(0,len(self.main_df)-MINUTES_PER_EPISODE-1)
		self.episode_df = self.main_df#[self.iloc:self.iloc+24]
		self.money_in = 'USD'
		self.start_time = self.episode_df['time'].iloc[self.iloc]
		self.end_time = self.episode_df['time'].iloc[self.iloc-1]
	def checkRatio(self, u):
		#print(self.ss)

		#print(self.net_worth)
		#print(self.balances['USDT'])
		#print(self.net_worth)
		#print(self.balances['USDT'])
		for s in self.ss[u]:
		   # print(s)
			if datetime.fromtimestamp(int(s['dt']['$date'] / 1000)) > thetime and s['id'] not in self.donetweets:# and datetime.fromtimestamp(int(s['dt']['$date'] / 1000)) < datetime.strptime(END_TIME,'%d %b, %Y'):
				
				ratio = s['s'].replace('#', '').replace('$', '')

				if ratio != 'USDT':# and f'{ratio}-USD_close' in self.episode_df:
		
				
						#try:
						if ratio != 'USDT':
							if s['score'] >= float(os.environ['sentiment']):
								
								#A=1#print (s['score'])
								
								if True: #datetime.strptime(self.episode_df[f'time'][self.iloc],'%Y-%m-%d %H:%M:%S')  > datetime.fromtimestamp(int(s['dt']['$date'] / 1000)):
									#A=1#print(s['dt'])
									#A=1#print(self.episode_df[f'time'][self.iloc])
									if s['id'] not in self.donetweets:
										print(ratio)
										#A=1#print(len(ss[u]))
										#self.ss[u].remove(s)
										#s['done'] = True
										#A=1#print(len(ss[u]))
										self.donetweets.append(s['id'])
										if True:#if self.balances[ratio] == 0:
									#if self.episode_df[f'{ratio}_{SMA_LOW}'][self.iloc] > self.episode_df[f'{ratio}_{SMA_HIGH}'][self.iloc] and self.episode_df[f'{ratio}_{SMA_LOW}'][self.iloc-1] > self.episode_df[f'{ratio}_{SMA_HIGH}'][self.iloc-1]:
											
											try:
											   # price = float(price)
												self.to_buy = ratio
												for exchange in exchanges:
													try:
														price = exchanges["31YHFx31DnLBvNfnohFxhLAnHZtsGrJnkbpbeswutJaetzdflS753Uqnj6pCUB61"].fetchTicker(ratio + "/USDT")['info']['lastPrice']	
														price = float(price)
													except:
														try:
															price = binspots.fetchTicker(ratio + "/USDT")['ask']	
															price = float(price)
														except:
															PrintException()
													#tobuyusd = ((self.balances[exchange]['USDT'] / float(os.environ['balancedivisor'])) * self.weights[exchange][u] ) / (len(self.stoplosses[exchange][self.to_buy])+1)
													#tobuy = tobuyusd / price
													#print(tobuyusd)
													#print(tobuy)
													#self.balance = self.balance - self.balances[self.to_buy]
													#print(self.balance)
													#self.balances['USD'] = 0.0
													self.buy_price = price
													A=1#print((len(self.thisbal[self.to_buy])+1))
													#A=1#print(self.balance)
													#self.balances['USD'] = 0.0
													
													if exchange not in self.buys:
														self.buys[exchange] = {}
													if exchange not in self.user_buys:
														self.user_buys[exchange] = {}

													if exchange not in self.stoplosses:
														self.stoplosses[exchange] = {}
													gogo = False	
													if self.to_buy not in self.stoplosses[exchange]:
														gogo = True
													elif len(self.stoplosses[exchange][self.to_buy]) == 0:
														gogo = True

													if gogo == True:
													#self.balances['USD'] -= self.weights[exchange][u] * (self.balances['USD']/float(os.environ['balancedivisor'])/(len(self.thisbal[self.to_buy])+1))
														self.user_buys[exchange][self.to_buy] = [u]
														self.buys[exchange][self.to_buy] = [price]
														self.stoplosses[exchange][self.to_buy] = [(price * float(os.environ['stop']))]
														print(self.stopsizes)
														self.stopsizes[exchange][self.to_buy] =  [(price * (1-float(os.environ['stop'])))]
														#self.thisbal[exchange][self.to_buy] = [float(tobuy)]
														#buy that ratio (self.to_buy)
														#self.balance = self.balance - self.balances[self.to_buy]

														#A=1#print(tweets2[s['id']].created_at)
														#A=1#print(tweets2[s['id']].full_text)
														memory.add_to_memory(f'Buy {self.to_buy}: {price} datetime: {s["ca"]}')
														self.money_in = self.to_buy
														print(memory.actions)
														
														print(datetime.fromtimestamp(int(s['dt']['$date'] / 1000)))
														#perc = (self.iloc2 / (MINUTES_PER_EPISODE)) * 100
														#perc = round(perc * 1000) / 1000
														A=1#print('We are ' + str(perc) + '% thru the backtest!')
														print('combined nw $' + str(self.net_worth))
														
														self.buysigs.append({'coin': self.to_buy, 'time': datetime.utcnow(), 'price': self.net_worth})
														self.returns.loc[self.rcount] = [datetime.fromtimestamp(int(s['dt']['$date'] / 1000))] + [self.net_worth]
														self.rcount = self.rcount + 1
														"""
														try:
															exchanges[exchange].createOrder (ratio + '/USDT', 'market', 'buy', tobuy, None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
														except Exception as e:
															print(e)
														"""
														qty = ((float(os.environ['balancedivisor'])) * self.weights[exchange][u] ) / (len(self.stoplosses[exchange][self.to_buy]))
														thecount = 1
														for s2 in self.ratios:

																if s2 in self.buys[exchange]:
																	if len(self.buys[exchange][s2]) > 0:
																		thecount = thecount + 1
														qty = min(qty, 92.5 / thecount)
														print('qty: ' + str(qty))
														#qty = 1 / qty
														#print(qty)
														#qty = qty * 100
														signalId = randomword(10) 
														if self.to_buy not in self.signalIds[exchange]:
															self.signalIds[exchange][self.to_buy] = [signalId]
														else:
															self.signalIds[exchange][self.to_buy].append(signalId)
														if self.to_buy not in self.signalIds2[exchange]:
															urll= "https://zignaly.com/api/signals.php?signalId="+ signalId + "&key=ddf42e3aad15cb4ee14af970b8c7e812&d95&positionSizePercentage=" + str(qty) + "&pair=" + (ratio + 'USDT').replace('/', '') + "&type=entry&leverage=3&side=long&exchange=binance&exchangeAccountType=futures&positionSizeQuote=USDT"
														
															self.signalIds2[exchange][self.to_buy] = [signalId]
														elif len(self.signalIds2[exchange][self.to_buy]) == 0:
															urll= "https://zignaly.com/api/signals.php?signalId="+ signalId + "&key=ddf42e3aad15cb4ee14af970b8c7e812&d95&positionSizePercentage=" + str(qty) + "&pair=" + (ratio + 'USDT').replace('/', '') + "&type=entry&leverage=3&side=long&exchange=binance&exchangeAccountType=futures&positionSizeQuote=USDT"
														
															self.signalIds2[exchange][self.to_buy] = [signalId]
														else:
															urll= "https://zignaly.com/api/signals.php?signalId="+ self.signalIds2[exchange][self.to_buy][0] + "&key=ddf42e3aad15cb4ee14af970b8c7e812&d95&DCAAmountPercentage1=50&pair=" + (ratio + 'USDT').replace('/', '') + "&type=DCA&leverage=3&DCATargetPercentage1=-6&side=long&exchange=binance&exchangeAccountType=futures&positionSizeQuote=USDT"
														print(urll)
														result = requests.get(urll)
														print(result)
														urll= "https://zignaly.com/api/signals.php?signalId="+ signalId + "&key=d021c235d6db8dc0c79109e610f937b7&d95&positionSizePercentage=" + str(qty) + "&pair=" + (ratio + 'USDT').replace('/', '') + "&type=entry&exchange=binance&exchangeAccountType=spot&positionSizeQuote=USDT"
							
														print(urll)
														result = requests.get(urll)
														print(result)
														urll= "https://zignaly.com/api/signals.php?signalId="+ signalId + "&key=9ed836d4a3572f92011cd070d6e85897&d95&positionSizePercentage=" + str(qty) + "&pair=" + (ratio + 'USDT').replace('/', '') + "&type=entry&exchange=kucoin&exchangeAccountType=spot&positionSizeQuote=USDT"
							
														print(urll)
														result = requests.get(urll)
														print(result)
														for futspot in signals:
															for signal in signals[futspot]:
																print(signal)
																if ratio == signal:
																	payload = {"id": signals[futspot][signal],
																			"action": "long_entry"}
																	print(payload)
																	url = "https://mudrex.com/api/v1/signals"
																	result = requests.post(url, json = payload)
																	print(result.text)
														sleep(2)
														with open ("ubs3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.user_buys))
														with open ("bs3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.buys))
														with open ("sls3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.stoplosses))
														with open ("stops3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.stopsizes))
														with open ("sigids32.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.signalIds2))
														with open ("sigids3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.signalIds))
														with open ("ids3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.donetweets))
														with open ("bals3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.thisbal))
												#break
												A=1#print(memory.actions)
											except ValueError:
												a=1
											except Exception as e:
												PrintException()
												A=1#print(e)
							if s['score'] <= -1 * float(os.environ['sentiment']):
								
								#A=1#print (s['score'])
								
								if True: #datetime.strptime(self.episode_df[f'time'][self.iloc],'%Y-%m-%d %H:%M:%S')  > datetime.fromtimestamp(int(s['dt']['$date'] / 1000)):
									#A=1#print(s['dt'])
									#A=1#print(self.episode_df[f'time'][self.iloc])
									if s['id'] not in self.donetweets:
										print(ratio)
										#A=1#print(len(ss[u]))
										#self.ss[u].remove(s)
										#s['done'] = True
										#A=1#print(len(ss[u]))
										self.donetweets.append(s['id'])
										if True:#if self.balances[ratio] == 0:
									#if self.episode_df[f'{ratio}_{SMA_LOW}'][self.iloc] > self.episode_df[f'{ratio}_{SMA_HIGH}'][self.iloc] and self.episode_df[f'{ratio}_{SMA_LOW}'][self.iloc-1] > self.episode_df[f'{ratio}_{SMA_HIGH}'][self.iloc-1]:
											
											try:
											   # price = exfloat(price)
												self.to_buy = ratio
												for exchange in exchanges:
													try:
														price = exchanges["31YHFx31DnLBvNfnohFxhLAnHZtsGrJnkbpbeswutJaetzdflS753Uqnj6pCUB61"].fetchTicker(ratio + "/USDT")['info']['lastPrice']	
														price = float(price)
													except:
														try:
															price = binspots.fetchTicker(ratio + "/USDT")['ask']	
															price = float(price)
														except:
															PrintException()
													#tobuyusd = ((self.balances[exchange]['USDT'] / float(os.environ['balancedivisor'])) * self.weights[exchange][u] ) / (len(self.stoplosses[exchange][self.to_buy])+1)
													#tobuy = tobuyusd / price
													#print(tobuyusd)
													#print(tobuy)
													#self.balance = self.balance - self.balances[self.to_buy]
													#print(self.balance)
													#self.balances['USD'] = 0.0
													self.buy_price = price
													A=1#print((len(self.thisbal[self.to_buy])+1))
													#A=1#print(self.balance)
													#self.balances['USD'] = 0.0
													if exchange not in self.buys:
														self.buys[exchange] = {}

													if exchange not in self.stoplosses:
														self.stoplosses[exchange] = {}
													gogo = False	
													if self.to_buy not in self.stoplosses[exchange]:
														gogo = True
													elif len(self.stoplosses[exchange][self.to_buy]) == 0:
														gogo = True

													if gogo == True:
														#self.balances['USD'] -= self.weights[exchange][u] * (self.balances['USD']/float(os.environ['balancedivisor'])/(len(self.thisbal[self.to_buy])+1))
														self.buys[exchange][self.to_buy] = [(price)]
														self.stoplosses[exchange][self.to_buy] = [(price * -1*float(os.environ['stop']))]
														print(self.stopsizes)
														self.stopsizes[exchange][self.to_buy] =  [(price * (1+float(os.environ['stop'])))]
														#self.thisbal[exchange][self.to_buy] = [float(tobuy)]
														#buy that ratio (self.to_buy)
														#self.balance = self.balance - self.balances[self.to_buy]

														#A=1#print(tweets2[s['id']].created_at)
														#A=1#print(tweets2[s['id']].full_text)
														memory.add_to_memory(f'Buy {self.to_buy}: {price} datetime: {s["ca"]}')
														self.money_in = self.to_buy
														print(memory.actions)
														
														print(datetime.fromtimestamp(int(s['dt']['$date'] / 1000)))
														#perc = (self.iloc2 / (MINUTES_PER_EPISODE)) * 100
														#perc = round(perc * 1000) / 1000
														A=1#print('We are ' + str(perc) + '% thru the backtest!')
														print('combined nw $' + str(self.net_worth))
														self.buysigs.append({'coin': self.to_buy, 'time': datetime.utcnow(), 'price': self.net_worth})
														self.returns.loc[self.rcount] = [datetime.fromtimestamp(int(s['dt']['$date'] / 1000))] + [self.net_worth]
														self.rcount = self.rcount + 1
														"""
														try:
															exchanges[exchange].createOrder (ratio + '/USDT', 'market', 'buy', tobuy, None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
														except Exception as e:
															print(e)
														"""
														qty = ((float(os.environ['balancedivisor'])) * self.weights[exchange][u] ) / (len(self.stoplosses[exchange][self.to_buy]))
														thecount = 1
														for s2 in self.ratios:

																if s2 in self.buys[exchange]:
																	if len(self.buys[exchange][s2]) > 0:
																		thecount = thecount + 1
														qty = min(qty, 92.5 / thecount)
														#qty = 1 / qty
														#print(qty)
														#qty = qty * 100
														signalId = randomword(10) 
														if self.to_buy not in self.signalIds[exchange]:
															self.signalIds[exchange][self.to_buy] = [signalId]
														else:
															self.signalIds[exchange][self.to_buy].append(signalId)
														if self.to_buy not in self.signalIds2[exchange]:
															self.signalIds2[exchange][self.to_buy] = [signalId]
															urll= "https://zignaly.com/api/signals.php?signalId="+ signalId + "&key=ddf42e3aad15cb4ee14af970b8c7e812&d95&positionSizePercentage=" + str(qty) + "&pair=" + (ratio + 'USDT').replace('/', '') + "&type=entry&leverage=3&side=short&exchange=binance&exchangeAccountType=futures&positionSizeQuote=USDT"
														elif len(self.signalIds2[exchange][self.to_buy]) == 0:
															self.signalIds2[exchange][self.to_buy] = [signalId]
															urll= "https://zignaly.com/api/signals.php?signalId="+ signalId + "&key=ddf42e3aad15cb4ee14af970b8c7e812&d95&positionSizePercentage=" + str(qty) + "&pair=" + (ratio + 'USDT').replace('/', '') + "&type=entry&leverage=3&side=short&exchange=binance&exchangeAccountType=futures&positionSizeQuote=USDT"
														else:
															urll= "https://zignaly.com/api/signals.php?signalId="+ self.signalIds2[exchange][self.to_buy][0] + "&key=ddf42e3aad15cb4ee14af970b8c7e812&d95&DCAAmountPercentage1=50&pair=" + (ratio + 'USDT').replace('/', '') + "&type=DCA&DCATargetPercentage1=6&leverage=3&side=short&exchange=binance&exchangeAccountType=futures&positionSizeQuote=USDT"
														print(urll)
														result = requests.get(urll)
														print(result)
														for futspot in signals:
															for signal in signals[futspot]:
																print(signal)
																if ratio == signal:
																	payload = {"id": signals[futspot][signal],
																			"action": "short_entry"}
																	print(payload)
																	url = "https://mudrex.com/api/v1/signals"
																	result = requests.post(url, json = payload)
																	print(result.text)
														sleep(2)
														with open ("ubs3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.user_buys))
														with open ("bs3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.buys))
														with open ("sls3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.stoplosses))
														with open ("stops3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.stopsizes))
														with open ("sigids32.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.signalIds2))
														with open ("sigids3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.signalIds))
														with open ("ids3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.donetweets))
														with open ("bals3.json", 'w') as slsjson:

															slsjson.write(json.dumps(self.thisbal))
												#break
												A=1#print(memory.actions)
											except ValueError:
												a=1
											except Exception as e:
												PrintException()
												A=1#print(e)
									
				   # except Exception as e:
					  #  A=1#print(e)
	def step(self):
		
		#-------IMPLEMENT STRATEGY HERE--------
		doneus = []
		#while len(doneus) < len(users):
		firstu = True

		if firstu == True:
			with open (os.environ['lorm'], "r") as f:
				try:
					self.ss = json.loads(f.read())
					#print(self.ss)
				except:
					abc=213
			firstu = False
	
		for exchange in exchanges:
			
			for s in self.stoplosses[exchange] :
				#print(s)
				try:
					#price = float(price)
					try:
						price = exchanges["31YHFx31DnLBvNfnohFxhLAnHZtsGrJnkbpbeswutJaetzdflS753Uqnj6pCUB61"].fetchTicker(s + "/USDT")['info']['lastPrice']
						price = float(price)
					except:
						try:
							price = binspots.fetchTicker(s + "/USDT")['ask']	
							price = float(price)
						except:
							PrintException()
					try:
						ubtemp = self.user_buys[exchange][s]
						btemp = self.buys[exchange][s]
						b2temp = self.buys[s]
						htemp = self.thishold[s]
						b3temp = self.thisbal[s]
					except:
						abc=123
					count = -1
					#if s == 'CHR':
						#print(self.stopsizes[exchange][s])
					for abuy in self.stoplosses[exchange][s]:
						count = count + 1
						abuy = int(abuy)
						#price = price 
						if self.stoplosses[exchange][s][count] > 0:
							if (price - self.stoplosses[exchange][s][count]) > self.stopsizes[exchange][s][count]:
								self.stopsizes[exchange][s][count] = price - self.stoplosses[exchange][s][count]
								print("New high observed, new stop size: " + str(self.stopsizes[exchange][s][count]) +" , now updating " + s + " stop loss to %.8f" % self.stoplosses[exchange][s][count] + ', net worth: ' + str(self.net_worth))
								self.stopsizes[exchange][s][count] = self.stopsizes[exchange][s][count] * 1.00025
								self.stoplosses[exchange][s][count] = self.stoplosses[exchange][s][count] / 1.00025
								with open ("sls3.json", 'w') as slsjson:

									slsjson.write(json.dumps(self.stoplosses))
								with open ("stops3.json", 'w') as slsjson:

									slsjson.write(json.dumps(self.stopsizes))
								
							elif price <= self.stopsizes[exchange][s][count]:
								#self.running = False
								#amount = self.binance.get_balance(self.market.split("/")[0])
								#price = self.binance.get_price(self.market)
								#self.binance.sell(self.market, amount, price)
								print("Sell triggered | Price: %.8f | Stop loss: %.8f" % (price, self.stoplosses[exchange][s][count]))
								try:
									if price > 1 * self.buys[exchange][s][count] or price < 1 * self.buys[exchange][s][count]: 
										try:
											self.weights[exchange][self.user_buys[exchange][s][count]] = self.weights[exchange][self.user_buys[exchange][s][count]] * (1 + (float(os.environ['learnwin']) / 100))
										except:
											PrintException()
										try:
											
											self.weights[exchange][self.user_buys[exchange][s][count]] = self.weights[exchange][self.user_buys[exchange][s][count]] * (1 - (float(os.environ['learnlose']) / 100))
										except:
											PrintException()
										self.weights[exchange][self.user_buys[exchange][s][count]] = min(self.weights[exchange][self.user_buys[exchange][s][count]], 3)
								except:
									PrintException()
								with open ("weights3.json", 'w') as slsjson:

									slsjson.write(json.dumps(self.weights))
								A=1#print(ubtemp[count] + ': ' + str(self.weights[exchange][ubtemp[count]]))
							#if self.episode_df[f'{self.money_in}_{SMA_LOW}'][self.iloc] < self.episode_df[f'{self.money_in}_{SMA_HIGH}'][self.iloc]:
								#if high sma crosses below low sma
								#sell money_in/USD
								A=1#print(s)
								A=1#print(htemp)
								A=1#print(self.balances[s])
								A=1#print(self.balances['USD'])
							   # self.balances[s] -= (htemp[count])
								#self.balances['USD'] += (htemp[count]*price)
								
								A=1#print(self.balances[s])
								A=1#print(self.balances['USD'])
								self.sell_price = price
								memory.add_to_memory(f'Sell {s}: {self.sell_price} datetime: {datetime.utcnow()}' )
								#A=1#print(memory.actions)
								
								
								perc = (self.iloc2 / (MINUTES_PER_EPISODE)) * 100
								perc = round(perc * 1000) / 1000
								A=1#print('Sold ' + s + '! We are ' + str(perc) + '% thru the backtest!')
								print('combined nw $' + str(self.net_worth))
								self.sellsigs.append({'coin': s, 'time': datetime.utcnow(), 'price': self.net_worth})
								#self.returns.loc[self.rcount] = [datetime.fromtimestamp(int(s['dt']['$date'] / 1000))] + [self.net_worth]
								self.rcount = self.rcount + 1

								#
								#self.thishold[s].remove(htemp[count])
								#self.buys[s].remove(b2temp[count])
								#self.user_buys[exchange][s].remove(ubtemp[count])
								#self.buys[exchange][s].remove(btemp[count])

								print(self.net_worth)
								
								"""
								try:
									exchanges[exchange].createOrder (s + '/USDT', 'market', 'sell', self.thisbal[exchange][s][count], None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
								except Exception as e:
									try:
										exchanges[exchange].createOrder (s + '/USDT', 'market', 'sell', self.balances[s], None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
									except Exception as e:
										print(e)
								"""   
								urll= "https://zignaly.com/api/signals.php?key=ddf42e3aad15cb4ee14af970b8c7e812&pair=" + s + "USDT&type=exit&exchange=binance&exchangeAccountType=futures&signalId=" + self.signalIds[exchange][s][count]
										
								print(urll)
								result = requests.get(urll)
								print(result)
								urll= "https://zignaly.com/api/signals.php?key=d021c235d6db8dc0c79109e610f937b7&pair=" + s + "USDT&type=sell&exchange=binance&exchangeAccountType=spot&signalId=" + self.signalIds[exchange][s][count]
										
								print(urll)
								result = requests.get(urll)
								print(result)
								urll= "https://zignaly.com/api/signals.php?key=9ed836d4a3572f92011cd070d6e85897&pair=" + s + "USDT&type=sell&exchange=kucoin&exchangeAccountType=spot&signalId=" + self.signalIds[exchange][s][count]
										
								print(urll)
								result = requests.get(urll)
								print(result)
								for futspot in signals:
									for signal in signals[futspot]:
										print(signal)
										if s == signal:
											payload = {"id": signals[futspot][signal],
													"action": "long_exit"}
											print(payload)
											url = "https://mudrex.com/api/v1/signals"
											result = requests.post(url, json = payload)
											print(result.text)

								sleep(2)
								try:
									self.user_buys[exchange][s].remove(self.user_buys[exchange][s][count])

									self.buys[exchange][s].remove(self.buys[exchange][s][count])
									with open ("ubs3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.user_buys))
									with open ("bs3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.buys))
								except:
									abc=123
								try:
									self.stoplosses[exchange][s].remove(self.stoplosses[exchange][s][count])
									self.stopsizes[exchange][s].remove(self. stopsizes[exchange][s][count])
									self.signalIds[exchange][s]. remove (self.signalIds[exchange][s][count])
									self.signalIds2[exchange][s] = []
									
									with open ("bals3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.thisbal))
									with open ("sls3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.stoplosses))
									with open ("stops3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.stopsizes))
									with open ("sigids32.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.signalIds2))
									with open ("sigids3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.signalIds))
								except:
									PrintException()
									sleep(10)

									abc=123
							#else:
								#print('trailtp no action..')
						else:
							if (price - self.stoplosses[exchange][s][count]) < self.stopsizes[exchange][s][count]:
								self.stopsizes[exchange][s][count] = price - self.stoplosses[exchange][s][count]
							   # print("New low observed, new stop size: " + str(self.stoplosses[exchange][s][count]) +" , now updating " + s + " stop loss to %.8f" % self.stoplosses[exchange][s][count] + ', net worth: ' + str(self.net_worth))
								
								if price < self.buys[exchange][s][count]: 
									self.stopsizes[exchange][s][count] = self.stopsizes[exchange][s][count] / 1.001
									self.stoplosses[exchange][s][count] = self.stoplosses[exchange][s][count] / 1.001
								print(print("New low observed, price: " + str(price) + " new stop size: " + str(self.stopsizes[exchange][s][count]) +" , now updating " + s + " stop loss to %.8f" % self.stoplosses[exchange][s][count] + ', net worth: ' + str(self.net_worth)))
								with open ("sls3.json", 'w') as slsjson:

									slsjson.write(json.dumps(self.stoplosses))
								with open ("stops3.json", 'w') as slsjson:

									slsjson.write(json.dumps(self.stopsizes))
								if 'XRP' in s:
									print("New low observed, price: " + str(price) + " new stop size: " + str(self.stopsizes[exchange][s][count]) +" , now updating " + s + " stop loss to %.8f" % self.stoplosses[exchange][s][count] + ', net worth: ' + str(self.net_worth))
							elif price >= self.stopsizes[exchange][s][count]:

								
								#self.running = False
								#amount = self.binance.get_balance(self.market.split("/")[0])
								#price = self.binance.get_price(self.market)
								#self.binance.sell(self.market, amount, price)
								print("Sell triggered | Price: %.8f | Stop loss: %.8f" % (price, self.stoplosses[exchange][s][count]))
							#if price > 1.1 * btemp[count] * self.weights[exchange][ubtemp[count]] or price < 0.94 * btemp[count]: 
								try:
									if price < 1 * self.buys[exchange][s][count] or price > 1 * self.buys[exchange][s][count]: 
										try:
											self.weights[exchange][self.user_buys[exchange][s][count]] = self.weights[exchange][self.user_buys[exchange][s][count]] * (1 + (float(os.environ['learnwin']) / 100))
										except:
											abc=123
										try:
											
											self.weights[exchange][self.user_buys[exchange][s][count]] = self.weights[exchange][self.user_buys[exchange][s][count]] * (1 - (float(os.environ['learnlose']) / 100))
										except:
											abc=123
										self.weights[exchange][self.user_buys[exchange][s][count]] = min(self.weights[exchange][self.user_buys[exchange][s][count]], 3)
								except:
									abc=123
								with open ("weights3.json", 'w') as slsjson:

									slsjson.write(json.dumps(self.weights))
								A=1#print(ubtemp[count] + ': ' + str(self.weights[exchange][ubtemp[count]]))
							#if self.episode_df[f'{self.money_in}_{SMA_LOW}'][self.iloc] < self.episode_df[f'{self.money_in}_{SMA_HIGH}'][self.iloc]:
								#if high sma crosses below low sma
								#sell money_in/USD
								A=1#print(s)
								A=1#print(htemp)
								A=1#print(self.balances[s])
								A=1#print(self.balances['USD'])
							   # self.balances[s] -= (htemp[count])
								#self.balances['USD'] += (htemp[count]*price)
								
								A=1#print(self.balances[s])
								A=1#print(self.balances['USD'])
								self.sell_price = price
								memory.add_to_memory(f'Sell {s}: {self.sell_price} datetime: {datetime.utcnow()}' )
								#A=1#print(memory.actions)
								
								
								perc = (self.iloc2 / (MINUTES_PER_EPISODE)) * 100
								perc = round(perc * 1000) / 1000
								A=1#print('Sold ' + s + '! We are ' + str(perc) + '% thru the backtest!')
								print('combined nw $' + str(self.net_worth))
								self.sellsigs.append({'coin': s, 'time': datetime.utcnow(), 'price': self.net_worth})
								#self.returns.loc[self.rcount] = [datetime.fromtimestamp(int(s['dt']['$date'] / 1000))] + [self.net_worth]
								self.rcount = self.rcount + 1

								#
								#self.thishold[s].remove(htemp[count])
								#self.buys[s].remove(b2temp[count])
								#self.user_buys[exchange][s].remove(ubtemp[count])
								#self.buys[exchange][s].remove(btemp[count])

								print(self.net_worth)
								
								"""
								try:
									exchanges[exchange].createOrder (s + '/USDT', 'market', 'sell', self.thisbal[exchange][s][count], None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
								except Exception as e:
									try:
										exchanges[exchange].createOrder (s + '/USDT', 'market', 'sell', self.balances[s], None, {"newClientOrderId": "x-v0tiKJjj-" + randomword(20)})
									except Exception as e:
										print(e)
								"""   
								urll= "https://zignaly.com/api/signals.php?key=ddf42e3aad15cb4ee14af970b8c7e812&pair=" + s + "USDT&type=exit&exchange=binance&exchangeAccountType=futures&signalId=" + self.signalIds[exchange][s][count]
										
								print(urll)
								result = requests.get(urll)
								print(result)
								for futspot in signals:
									for signal in signals[futspot]:
										print(signal)
										if s == signal:
											payload = {"id": signals[futspot][signal],
													"action": "short_exit"}
											print(payload)
											url = "https://mudrex.com/api/v1/signals"
											result = requests.post(url, json = payload)
											print(result.text)
								sleep(2)
								try:
									self.user_buys[exchange][s].remove(self.user_buys[exchange][s][count])

									self.buys[exchange][s].remove(self.buys[exchange][s][count])
									with open ("ubs3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.user_buys))
									with open ("bs3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.buys))
								except:
									abc=123
								try:
									self.stoplosses[exchange][s].remove(self.stoplosses[exchange][s][count])
									self.stopsizes[exchange][s].remove(self. stopsizes[exchange][s][count])
									self.signalIds2[exchange][s] = []
									self.signalIds[exchange][s]. remove (self.signalIds[exchange][s][count])
									with open ("bals3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.thisbal))
									with open ("sls3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.stoplosses))
									with open ("stops3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.stopsizes))
									with open ("sigids32.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.signalIds2))
									with open ("sigids3.json", 'w') as slsjson:

										slsjson.write(json.dumps(self.signalIds))
								except:
									abc=123
						#else:
							#print('price lower than current target!')
				except ValueError:
					a=1
				except Exception as e:
					A=1#print(e)
					PrintException()
								#self.money_in = 'USD'
		for u in self.ss:
			for exchange in exchanges:
				if u not in self.weights[exchange]:
					self.weights[exchange][u] = 1
			self.balances = {}
			self.net_worth = 0
			for exchange in exchanges:
				self.balances[exchange] = {}
				sleeps = []
				"""
				try:
					bal = exchanges[exchange].fetchBalance()
					#print(bal)
					for b in bal:
						if b == 'total':
							for coin in bal[b]:
								self.balances[exchange][coin] = bal[b][coin]

					self.returns.loc[self.rcount] = [datetime.utcnow()] + [self.net_worth]
					
					try:
						self.net_worth += self.balances[exchange]['USDT']
					except:
						print(bal)
				except:
					abc=123
				"""
				
			#print('combined nw $' + str(self.net_worth))
			#sleep(100)
			self.rcount = self.rcount + 1
			self.iloc2 += 1
			self.iloc+=1
		   #if self.iloc <= MINUTES_PER_EPISODE:
			#print(self.iloc)
	#if low sma crosses above high sma
	#try:
			if u in self.ss:
				#print(self.ss[u])
				
				if len(self.ss[u]) > 0:
					#print(len(self.ss[u]))
					if True:# datetime.strptime(self.episode_df[f'time'][self.iloc],'%Y-%m-%d %H:%M:%S')  > datetime.fromtimestamp(int(ss[u][-1]['dt']['$date'] / 1000)):
					   # A=1#print(self.iloc2)

						
						
						
						self.checkRatio(u)
						

					else:
						doneus.append(u)
				else:
					doneus.append(u)
			else:
				doneus.append(u)
		#except:
				#	a=1

		#-------IMPLEMENT STRATEGY HERE--------
		
		#-------CALCULATE PERFORMANCE METRICS HERE-------
		#Running net worth
		
		return self.buysigs, self.sellsigs, self.returns, self.net_worth, 1, self.start_time, self.end_time

#select cryptocurrencies you'd like to gather and time interval
merge = False
import os.path
from os import path
df = pd.read_csv("minutely")
"""
thetime = None
for t in df['time']:
	atime = datetime.strptime(t,'%Y-%m-%d %H:%M:%S')
	if thetime == None:
		thetime = atime
	elif atime > thetime:
		thetime = atime
	else:
		print(atime)
		print(thetime)
"""
#print(df['time'])
env = Env(ratios, df)
memory = Memory()

net_worth_collect = []
average_market_change_collect = []
net_worth = 0
sentiments = [0.5, 0.6, 0.7, 0.8]
stops = [0.95, 0.85, 0.8, 0.7]
lorms = ['more', 'less']
learnwins = [1,5, 9]
learnloses = [1, 5, 9]
import os
ss = []
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
analyser = SentimentIntensityAnalyzer() 
new_words = {
'bullish': 0.75, 
'bearish': -0.75,
'neutral': 0,
}
analyser.lexicon.update(new_words) 
def sentiment_analyzer_scores(sentence):
	score = analyser.polarity_scores(sentence)
	return(score)
while True:
	dos = ['DOGE', 'ETH', 'BTC', 'MARS', 'BCH', 'BNB', 'ADA', 'XRP', 'LTC', 'XLM', 'XEM', 'AAVE', 'EOS', 'BSV', 'XMR', 'TRX', 'FTT', 'XTZ', 'SNX', 'MKR', 'FIL', 'XSM']
	
	#print(ss)
	from time import sleep
	from bson import json_util
	#json.loads(aJsonString, object_hook=json_util.object_hook)
	import random
	rs = []

	dfdays = pd.DataFrame({},columns=['time', 'equity'])
	for i_episode in range(NUM_EPISODES):
		A=1#print(len(env.episode_df))
		temp = env.step()
		if temp != None:
			buysigs, sellsigs, df, tnet_worth, average_market_change, start_time, end_time = temp
		if tnet_worth != 0:
			net_worth = tnet_worth
	   
	with open (os.environ['lorm'], "r") as f:
		try:
			ss = json.loads(f.read())
		except:
			abc=123
	countabc = -1
	"""
	for u in users:
		countabc += 1
		#print(countabc)
		#u = "elonmusk"
		try:
		
			if ss[u] == None:
			
				ss[u] = []
		except:
			ss[u] = []
			
		recheck = True
		themax = 0
		for obj in ss[u]:
			if obj['id'] < themax or themax == 0:
				themax = obj['id']
		thedt = None  
		while recheck == True:
			try:
				anewdt = datetime.utcnow().timestamp()
				#print(anewdt)
				if themax == 0:
					
					tweets = api.user_timeline(include_rts=False,screen_name=u, count=200, tweet_mode='extended')#u.screen_name, count=10)
				else:
					tweets = api.user_timeline(max_id=themax, include_rts=False,screen_name=u, count=20, tweet_mode='extended')#u.screen_name, count=10)
				
				#tweets = [i.AsDict() for i in t]
			   # print(len(tweets))
				if len(tweets) == 0:
					recheck = False
				for t in tweets:
					if t.id < themax or themax == 0:
						themax = t.id - 1
						thedt = t.created_at
					if t.created_at <= threeyears:
						recheck = False
					pipelineModel = nlpPipeline.fit(empty_df)

					df = spark.createDataFrame(pd.DataFrame({"text":[t.full_text]}))
					result = pipelineModel.transform(df)

					#
				
					sarcasmis = result.first()['sentiment'][0].__getattr__("result")
					if sarcasmis == 'sarcasm':
						print(u + ' being sarcastic: ' + t.full_text)
					gogo = True
					#print(t.created_at)
					#ts = time.strptime(t.created_at,'%a %b %d %H:%M:%S +0000 %Y')
					dt = t.created_at 
					if dt > thetime:
						split=(t.full_text).replace('\n', ' ').split(' ')
						for s in split:
							#s = s.replace('$','')
							ago = False
							
							if ((('$' in s or '#' in s) and s.upper().replace('$','').replace('#','') in ratios) or ((s.upper().replace('$','') in ratios and s.upper().replace("$","").replace("#","") not in dontdo) and (s.upper().replace("$","").replace("#","") in ratios and s.upper().replace("$","").replace("#","") in dos))):
								ago = True
								#print(sentiment_analyzer_scores(t.full_text)['compound'] )
								if sentiment_analyzer_scores(t.full_text)['compound'] >= 0.0:
									t.full_text = t.full_text.replace('"', "'")
																			   
									ss[u].append({'ca': t.created_at, 'dt': dt, 'id': t.id, 's': s.upper().replace('$',''), 'score': sentiment_analyzer_scores(t.full_text)['compound']})
									
									with open(os.environ['lorm'], "w") as f:
										f.write(json.dumps(ss, default=json_util.default))
									##print(u)
									#print(s)
							elif 'less' in os.environ['lorm'] and (('$' in s or '#' in s) and s.upper().replace('$','').replace('#','') in cs):
								ago = True
								print('less')
								if sentiment_analyzer_scores(t.full_text)['compound'] > 0.2:
									t.full_text = t.full_text.replace('"', "'")
																			   
									ss[u].append({'ca': t.created_at, 'dt': dt, 'id': t.id, 's': s.upper().replace('$',''), 'score': sentiment_analyzer_scores(t.full_text)['compound']})
									
									with open(os.environ['lorm'], "w") as f:
										f.write(json.dumps(ss, default=json_util.default))
									print(s)
								#if (s.upper() in dos):
									#print(sentiment_analyzer_scores(t.full_text)['compound'])  
				anewdt2 = datetime.utcnow().timestamp()				
				diff = anewdt2 - anewdt
				#print(diff)
				diff = 0.61 #- diff
				#print(diff)
				sleep(diff)
				
			except Exception as
			 e:
				#print(e)
				if 'Rate limit' in str(e):
					sleep(60)
				recheck = False
		
		#if thedt != None:
			#print(u + ': ' + str(themax) + ' at ' + datetime.strftime(thedt, '%Y-%m-%d %H:%M:%S'))
	"""
	sleep(5)
	#log overall
	#A=1#print(f'net worth average after {NUM_EPISODES} backtest episodes: {np.mean(net_worth_collect)}')
	#Yes, average of the average market changes
	#A=1#print(f'average, average market change over {NUM_EPISODES} episodes: {np.mean(average_market_change_collect)}')
	#sleep(60)
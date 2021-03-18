const fetch = require("node-fetch");

var spots = []
var futures = []
var us = []
var spots2 = []
var futures2 =[]
var us2 = []
var futbals = []
var spotbals = []
var spotpercs = []
var futpercs = []
var uspercs = []
var usbals = []
var totalbals = []
var futbal = 0
var spotbal = 0
var usbal = 0
function doFetch(){
total = 0
var thetime = new Date().getTime()
 
    fetch("https://zignaly.com/api/fe/api.php?action=getBalanceAndPositionsForService", {
  "headers": {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9,lb;q=0.8",
    "content-type": "application/json",
    "sec-ch-ua": "\"Google Chrome\";v=\"89\", \"Chromium\";v=\"89\", \";Not A Brand\";v=\"99\"",
    "sec-ch-ua-mobile": "?0",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-api-key": "",
    "cookie": "ajs_anonymous_id=%22815ee61e-7a16-48b6-8fef-7a46a1c3f4e5%22; utm_source=; utm_campaign=; utm_medium=; utm_term=; utm_content=; ref=; ajs_user_id=%225f345c373a81961922505342%22; _gid=GA1.2.726638202.1615211808; amplitude_id_fef1e872c952688acd962d30aa545b9ezignaly.com=eyJkZXZpY2VJZCI6IjM2NTBjYTliLTA3NWItNDI3My1hMWU2LTUyMjQ5MzAwOThkOFIiLCJ1c2VySWQiOm51bGwsIm9wdE91dCI6ZmFsc2UsInNlc3Npb25JZCI6MTYxNTUxODMyODgyNCwibGFzdEV2ZW50VGltZSI6MTYxNTUxODMyOTMzOSwiZXZlbnRJZCI6MSwiaWRlbnRpZnlJZCI6MSwic2VxdWVuY2VOdW1iZXIiOjJ9; _ga=GA1.1.1443071867.1614971189; _ga_N47YM5S430=GS1.1.1615583929.48.1.1615584069.0"
  },
  "referrer": "https://zignaly.com/app/profitSharing/604489b1f76f99517a7211ba/management",
  "referrerPolicy": "same-origin",
  "body": "{\"token\":\"bb5fbf72cd88b10dc8b2635e1681c319\",\"providerId\":\"6051dd31ab6a996500410aa6\"}",
  "method": "POST",
  "mode": "cors"
}).then(async function(response) {
  return response.json().then(function(json) {
    console.log(json)
      futpercs.push([thetime, 100+(-1*json.balance.abstractPercentage)])
    
  fetch("https://zignaly.com/api/fe/api.php?action=getBalanceForService", {
  "headers": {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/json",
    "sec-ch-ua": "\"Google Chrome\";v=\"89\", \"Chromium\";v=\"89\", \";Not A Brand\";v=\"99\"",
    "sec-ch-ua-mobile": "?0",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-api-key": "",
    "cookie": "ajs_anonymous_id=%22b0fc0b4d-d95d-4300-b5bf-eeeaf82680c2%22; _gid=GA1.2.335720653.1615499267; utm_medium=; utm_source=; utm_campaign=; utm_term=; utm_content=; ref=; ajs_user_id=%226048dac0b28b96719551f713%22; _ga=GA1.1.332284921.1615499267; _ga_N47YM5S430=GS1.1.1615593191.10.1.1615594763.0"
  },
  "referrer": "https://zignaly.com/app/profitSharing/6051dd31ab6a996500410aa6/management",
  "referrerPolicy": "same-origin",
  "body": "{\"token\":\"bb5fbf72cd88b10dc8b2635e1681c319\",\"providerId\":\"6051dd31ab6a996500410aa6\"}",
  "method": "POST",
  "mode": "cors"
}).then(async function(response) {
  return response.json().then(function(json) {
    console.log(json)
    futbal = (json['totalMarginUSDT'])
    total += parseFloat(futbal)
  
fetch("https://zignaly.com/api/fe/api.php?action=getProviderList2", {
  "headers": {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/json",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-api-key": "",
    "cookie": "ajs_anonymous_id=%22b803d09a-0d4b-466e-90d3-bdc25284ecf4%22; _gid=GA1.2.1269233896.1615403787; ajs_user_id=%226048dac0b28b96719551f713%22; __ls_exp=1615399192; amplitude_id_fef1e872c952688acd962d30aa545b9ezignaly.com=eyJkZXZpY2VJZCI6ImMyNTVlMTlmLTcwMGEtNDI4Ny05ZjIzLWI2ODk1MTk3ZDJmMlIiLCJ1c2VySWQiOm51bGwsIm9wdE91dCI6ZmFsc2UsInNlc3Npb25JZCI6MTYxNTQ4NjU5ODQxNSwibGFzdEV2ZW50VGltZSI6MTYxNTQ4NjU5OTAzNCwiZXZlbnRJZCI6MSwiaWRlbnRpZnlJZCI6MSwic2VxdWVuY2VOdW1iZXIiOjJ9; utm_source=; utm_medium=; utm_campaign=; utm_term=; utm_content=; ref=; _ga=GA1.1.408316642.1615403787; _gat=1; _ga_N47YM5S430=GS1.1.1615507660.12.1.1615510171.0"
  },
  "referrer": "https://zignaly.com/app/dashboard/connectedTraders",
  "referrerPolicy": "same-origin",
  "body": "{\"token\":\"bb5fbf72cd88b10dc8b2635e1681c319\",\"type\":\"connected\",\"ro\":true,\"provType\":[\"copytraders\",\"profitsharing\"],\"timeFrame\":90,\"internalExchangeId\":\"Zignaly1615408769_60492e81299bf\",\"version\":3}",
  "method": "POST",
  "mode": "cors"
}).then(async function(response) {
	return response.json().then(function(json) {
		var returns = (json[0]['dailyReturns'])
		futures = []
		for (var r in returns){
			if (returns[r].returns != 0){
				futures.push([new Date(returns[r].name).getTime(), returns[r].returns])
			}
		}
    if (futures.length == 0){
      futures.push([new Date().getTime(), 0])
    }

fetch("https://zignaly.com/api/fe/api.php?action=getOpenPositions", {
  "headers": {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/json",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-api-key": "",
    "cookie": "ajs_anonymous_id=%22b803d09a-0d4b-466e-90d3-bdc25284ecf4%22; _gid=GA1.2.1269233896.1615403787; ajs_user_id=%226048dac0b28b96719551f713%22; __ls_exp=1615399192; amplitude_id_fef1e872c952688acd962d30aa545b9ezignaly.com=eyJkZXZpY2VJZCI6ImMyNTVlMTlmLTcwMGEtNDI4Ny05ZjIzLWI2ODk1MTk3ZDJmMlIiLCJ1c2VySWQiOm51bGwsIm9wdE91dCI6ZmFsc2UsInNlc3Npb25JZCI6MTYxNTQ4NjU5ODQxNSwibGFzdEV2ZW50VGltZSI6MTYxNTQ4NjU5OTAzNCwiZXZlbnRJZCI6MSwiaWRlbnRpZnlJZCI6MSwic2VxdWVuY2VOdW1iZXIiOjJ9; utm_source=; utm_medium=; utm_campaign=; utm_term=; utm_content=; ref=; _ga=GA1.1.408316642.1615403787; october_session=eyJpdiI6IitQNGU4RHVmcEQrZWFcL0V3bVpheHZ3PT0iLCJ2YWx1ZSI6IkxEYXFva2trSG1Ja1pXM1NPcWtrNVg0TDBWVnYyVEhEak51YVo5TzNLcU1McXBYQlhsSGx6YUpOTTloR2xjMXIiLCJtYWMiOiIzZDIxZmMxNjNmMTViM2U2YzcwZjVjNGVmNWUyMWQwMmNhNjRjNGMyYjdkYTJkM2FkNDhkNWRmZWJiYmE5YzUzIn0%3D; _ga_N47YM5S430=GS1.1.1615507660.12.1.1615513374.0"
  },
  "referrer": "https://zignaly.com/app/dashboard",
  "referrerPolicy": "same-origin",
  "body": "{\"token\":\"bb5fbf72cd88b10dc8b2635e1681c319\",\"internalExchangeId\":\"Zignaly1615408769_60492e81299bf\",\"version\":2}",
  "method": "POST",
  "mode": "cors"
}).then(async function(response) {
  return response.json().then(function(json) {
var upnl = 0
    for (var pos in json){
      upnl += json[pos].uPnL
    }
    total += upnl
 if (futbals.length == 0){
      futbals.push([thetime, futbal += upnl])
    }
      futbals.push([thetime, futbal += upnl])

        if (totalbals.length == 0){
      totalbals.push([thetime, total])
    }
      totalbals.push([thetime, total])

var bal = futbals[futbals.length-1][1]
    var diff = (upnl / bal) * 100



    futures[futures.length-1][1] += diff
    if (futures2.length == 0){
futures2.push([new Date().getTime(), futures[futures.length-1][1]])
    }
    if (futures2[futures2.length-1][1] != futures[futures.length-1][1]){
    futures2.push([new Date().getTime(), futures[futures.length-1][1]])
  }

   setTimeout(function(){
   doFetch()
 }, 100)
  })
console.log(12)
})

console.log(18)
  })
console.log(17)
})
console.log(16)
})
console.log(15)
})
console.log(14)
})
console.log(13)
})
}
doFetch()
setInterval(function(){
console.log(futures)
console.log(futures2)
}, 5000)

const express = require('express');
var cors = require('cors');
var app = express();
app.use(cors());
var bodyParser = require('body-parser')
app.use(bodyParser.json()); // to support JSON-encoded bodies
app.use(bodyParser.urlencoded({ // to support URL-encoded bodies
    extended: true
}));
app.set('view engine', 'ejs');
app.listen(process.env.PORT || 80, function() {});


app.get('/update', cors(), (req, res) => {
var tempspots = spots2
var tempfuts = futures2
var tempus = us2
if (spots2.length == 0){

tempspots = [[new Date().getTime(), 0]]
}
if (futures2.length == 0){
  tempfuts = [[new Date().getTime(), 0]]
}
if (us2.length == 0){
  tempus = [[new Date().getTime(), 0]]
}


	res.json({


spots: tempspots[tempspots.length-1],
futures:  tempfuts[tempfuts.length-1],
us:  tempus[tempus.length-1],
futbals:  futbals[futbals.length-1],
usbals:  usbals[usbals.length-1],
totalbals:  totalbals[totalbals.length-1],
spotbals:  spotbals[spotbals.length-1],
uspercs:  uspercs[uspercs.length-1],
futpercs:  futpercs[futpercs.length-1],
spotpercs:  spotpercs[spotpercs.length-1]
    })
})

app.get('/', cors(), (req, res) => {
  var tempspots = spots2
var tempfuts = futures2
var tempus = us2
if (spots2.length == 0){

tempspots = [[new Date().getTime(), 0]]
}
if (futures2.length == 0){
  tempfuts = [[new Date().getTime(), 0]]
}
if (us2.length == 0){
  tempus = [[new Date().getTime(), 0]]
}

console.log(tempus)
        res.render('index.ejs', {


spots: tempspots,
futures:  tempfuts,
us:  tempus,
futbals:  futbals,
uspercs:  uspercs,
spotpercs:  spotpercs,
futpercs: futpercs,
usbals:  usbals,
spotbals:  spotbals,
totalbals: totalbals
    })
})
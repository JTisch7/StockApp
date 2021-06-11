var stocks = JSON.parse('{{ stocks | tojson | safe}}');  
anychart.onDocumentReady(function () {

    // set the data
    table = anychart.data.table('date');
    //table.addData([{'opn': 43.58, 'high': 43.7275, 'low': 43.4875, 'close': 43.5725, 'date': '2019-02-28 06:30:00-08:00'}, {'opn': 43.5713, 'high': 43.5713, 'low': 43.4325, 'close': 43.4625, 'date': '2019-02-28 07:00:00-08:00'}, {'opn': 43.4625, 'high': 43.5477, 'low': 43.3975, 'close': 43.43, 'date': '2019-02-28 07:30:00-08:00'}]);
    //var stocks = JSON.parse('{{ stocks | tojson | safe}}');
    table.addData(stocks);

      
    // map the data
    mapping = table.mapAs({'open':"opn",'high': "high", 'low':"low", 'close':"close", 'volume':"volume"});
    chart = anychart.stock();
    
    // set the series
    var series = chart.plot(0).ohlc(mapping);
    series.name("{{ stk }} stock prices");

    // set the container id
    chart.container('container');

    // draw the chart
    chart.draw();

});
var stocksPred = JSON.parse('{{ stocksPred | tojson | safe}}');  
anychart.onDocumentReady(function () {

    // set the data
    table = anychart.data.table('date');
    //table.addData([{'opn': 43.58, 'high': 43.7275, 'low': 43.4875, 'close': 43.5725, 'date': '2019-02-28 06:30:00-08:00'}, {'opn': 43.5713, 'high': 43.5713, 'low': 43.4325, 'close': 43.4625, 'date': '2019-02-28 07:00:00-08:00'}, {'opn': 43.4625, 'high': 43.5477, 'low': 43.3975, 'close': 43.43, 'date': '2019-02-28 07:30:00-08:00'}]);
    //var stocks = JSON.parse('{{ stocks | tojson | safe}}');
    table.addData(stocksPred);

      
    // map the data
    mapping = table.mapAs({'open':"opn",'high': "high", 'low':"low", 'close':"close", 'volume':"volume"});
    chart = anychart.stock();
    
    // set the series
    var series = chart.plot(0).ohlc(mapping);
    series.name("{{ stockPred }} stock prices");

    // set the container id
    chart.container('cont');

    // draw the chart
    chart.draw();

});
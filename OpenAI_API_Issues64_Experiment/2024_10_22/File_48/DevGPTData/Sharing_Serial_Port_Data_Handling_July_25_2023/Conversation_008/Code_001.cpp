void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    heartRateData.append(data);
    heartRateTime.append(key);

    // Remove data that's more than 10 seconds old
    while (!heartRateTime.isEmpty() && key - heartRateTime.first() > 10) {
        heartRateTime.removeFirst();
        heartRateData.removeFirst();
    }

    // Adjust the x-axis range to make the data scroll
    if (key > heartRatePlot->xAxis->range().upper) {
        heartRatePlot->xAxis->setRange(key - 10, key);
    }

    heartRatePlot->graph(0)->setData(heartRateTime, heartRateData);
    heartRatePlot->replot();
}

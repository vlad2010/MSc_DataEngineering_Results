void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    heartRateData.append(data);
    heartRateTime.append(key);

    // If this is the first data point, save its time
    if (heartRateTime.size() == 1) {
        firstHeartRateTime = key;
    }

    // Adjust the x-axis range to make the data scroll 10 seconds after the first data point
    if (key - firstHeartRateTime > 10) {
        heartRatePlot->xAxis->setRange(key - 10, key);
    }

    heartRatePlot->graph(0)->setData(heartRateTime, heartRateData);
    
    // Auto-adjust the y-axis
    heartRatePlot->yAxis->rescale();

    heartRatePlot->replot();
}

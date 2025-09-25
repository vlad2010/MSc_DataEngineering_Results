void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    heartRateData.append(data);
    heartRateTime.append(key);

    // Remove data that's more than 10 seconds old, or beyond the 10th data point
    while (heartRateTime.size() > 10 && key - heartRateTime.first() > 10) {
        heartRateTime.removeFirst();
        heartRateData.removeFirst();
    }

    // Adjust the x-axis range to make the data scroll after 10 data points
    if (heartRateTime.size() > 10 && key > heartRatePlot->xAxis->range().upper) {
        heartRatePlot->xAxis->setRange(key - 10, key);
    }

    heartRatePlot->graph(0)->setData(heartRateTime, heartRateData);
    
    // Auto-adjust the y-axis
    heartRatePlot->yAxis->rescale();

    heartRatePlot->replot();
}

void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    heartRateData.append(data);
    heartRateTime.append(key);

    // Assuming the heart rate data comes at a frequency of 1Hz
    // Adjust the size of data to keep only the last 10 seconds of data (10 points)
    while (heartRateTime.size() > 10) {
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

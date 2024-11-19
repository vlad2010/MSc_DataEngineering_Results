void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    heartRateData.append(data);
    heartRateTime.append(key);

    heartRatePlot->graph(0)->setData(heartRateTime, heartRateData);
    heartRatePlot->xAxis->rescale();
    heartRatePlot->yAxis->rescale();
    heartRatePlot->replot();

    // Update the latest heart rate label
    latestHeartRateLabel->setText(QString("Latest Heart Rate: %1 bpm").arg(data));
}

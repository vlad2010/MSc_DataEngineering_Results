void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    heartRateData.append(data);
    heartRateTime.append(key);

    heartRatePlot->graph(0)->setData(heartRateTime, heartRateData);
    
    // Auto-adjust both the x-axis and the y-axis
    heartRatePlot->xAxis->rescale();
    heartRatePlot->yAxis->rescale();

    heartRatePlot->replot();
}

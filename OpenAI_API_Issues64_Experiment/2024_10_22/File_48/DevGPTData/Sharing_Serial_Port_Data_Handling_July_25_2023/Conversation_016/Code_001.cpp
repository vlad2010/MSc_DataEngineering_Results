void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    heartRateData.append(data);
    heartRateTime.append(key);

    heartRatePlot->graph(0)->setData(heartRateTime, heartRateData);
    
    // Adjust both axes with a margin
    heartRatePlot->xAxis->setRange(*std::min_element(heartRateTime.constBegin(), heartRateTime.constEnd()), 
                                   *std::max_element(heartRateTime.constBegin(), heartRateTime.constEnd()) + 1);
    heartRatePlot->yAxis->setRange(*std::min_element(heartRateData.constBegin(), heartRateData.constEnd()) - 1,
                                   *std::max_element(heartRateData.constBegin(), heartRateData.constEnd()) + 1);

    heartRatePlot->replot();
}

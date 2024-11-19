void MainWindow::addECGData(int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    ecgData.append(data);
    ecgTime.append(key);

    if (ecgTime.size() > 360) {
        ecgTime.removeFirst();
        ecgData.removeFirst();
    }

    ecgPlot->graph(0)->setData(ecgTime, ecgData);
    ecgPlot->replot();
}

void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    heartRateData.append(data);
    heartRateTime.append(key);

    heartRatePlot->graph(0)->setData(heartRateTime, heartRateData);
    heartRatePlot->replot();
}

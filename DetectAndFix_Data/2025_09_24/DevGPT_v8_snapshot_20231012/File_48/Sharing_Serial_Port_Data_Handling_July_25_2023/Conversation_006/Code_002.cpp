void MainWindow::addECGData(int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    ecgData.append(data);
    ecgTime.append(key);

    // Adjust the size of data to keep only the last 10 seconds of data (3600 points)
    while (ecgTime.size() > 3600) {
        ecgTime.removeFirst();
        ecgData.removeFirst();
    }

    // Adjust the x-axis range to make the data scroll
    if (key > ecgPlot->xAxis->range().upper) {
        ecgPlot->xAxis->setRange(key - 10, key);
    }

    ecgPlot->graph(0)->setData(ecgTime, ecgData);
    ecgPlot->replot();
}

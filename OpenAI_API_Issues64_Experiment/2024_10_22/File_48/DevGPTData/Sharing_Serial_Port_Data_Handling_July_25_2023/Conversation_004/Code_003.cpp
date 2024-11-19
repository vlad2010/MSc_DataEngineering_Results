void MainWindow::addECGData(int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    ecgData.append(data);
    ecgTime.append(key);

    if (ecgTime.size() > 360) {
        ecgTime.removeFirst();
        ecgData.removeFirst();
    }

    // Adjust the x-axis range to make the data scroll
    if (key > ecgPlot->xAxis->range().upper) {
        ecgPlot->xAxis->setRange(key - 1, key);
    }

    ecgPlot->graph(0)->setData(ecgTime, ecgData);
    ecgPlot->replot();
}

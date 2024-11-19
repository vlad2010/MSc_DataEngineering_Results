void MainWindow::addECGData(int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;
    
    // Reset the plot if we've reached the end of the window
    if (key > 1) {
        key = 0;
        startTime = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0;
        ecgTime.clear();
        ecgData.clear();
    }

    ecgData.append(data);
    ecgTime.append(key);

    ecgPlot->graph(0)->setData(ecgTime, ecgData);
    ecgPlot->replot();
}

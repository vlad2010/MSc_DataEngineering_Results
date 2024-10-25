void MainWindow::addHeartRateData(unsigned int data)
{
    ...
    // Update the label
    heartRateLabel->position->setType(QCPItemPosition::ptPlotCoords); // This line is important
    heartRateLabel->position->setCoords(key, data);
    heartRateLabel->setText(QString::number(data));
    heartRateLabel->setFont(QFont(font().family(), 8));
    heartRateLabel->setColor(Qt::black);
    heartRateLabel->setPadding(QMargins(1, 1, 1, 1));

    heartRatePlot->replot();
}

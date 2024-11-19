void MainWindow::addHeartRateData(unsigned int data)
{
    double key = QDateTime::currentDateTime().toMSecsSinceEpoch() / 1000.0 - startTime;

    heartRateData.append(data);
    heartRateTime.append(key);

    heartRatePlot->graph(0)->setData(heartRateTime, heartRateData);
    heartRatePlot->xAxis->setRange(*std::min_element(heartRateTime.constBegin(), heartRateTime.constEnd()),
                                   *std::max_element(heartRateTime.constBegin(), heartRateTime.constEnd()) + 1);
    heartRatePlot->yAxis->setRange(*std::min_element(heartRateData.constBegin(), heartRateData.constEnd()) - 1,
                                   *std::max_element(heartRateData.constBegin(), heartRateData.constEnd()) + 1);

    // Update the label
    heartRateLabel->position->setType(QCPItemPosition::ptPlotCoords);
    heartRateLabel->position->setCoords(key, data);
    heartRateLabel->setText(QString::number(data));
    heartRateLabel->setFont(QFont(font().family(), 8));
    heartRateLabel->setColor(Qt::black);
    heartRateLabel->setPadding(QMargins(1, 1, 1, 1));
    heartRateLabel->setPositionAlignment(Qt::AlignLeft|Qt::AlignVCenter);
    heartRateLabel->position->setPixelPosition(heartRateLabel->position->pixelPosition() + QPoint(10, 0));  // Add this line

    heartRatePlot->replot();
}

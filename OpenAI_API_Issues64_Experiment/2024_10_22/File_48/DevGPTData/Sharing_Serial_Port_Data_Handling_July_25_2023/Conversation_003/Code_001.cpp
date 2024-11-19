#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "qcustomplot.h"
#include "serialportreader.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void addECGData(int data);
    void addHeartRateData(unsigned int data);

private:
    QCustomPlot *ecgPlot;
    QCustomPlot *heartRatePlot;

    QVector<double> ecgData;
    QVector<double> heartRateData;

    QVector<double> ecgTime;
    QVector<double> heartRateTime;

    double startTime;

    SerialPortReader *serialReader;
};

#endif // MAINWINDOW_H

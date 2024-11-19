#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QVector>
#include <QDateTime>
#include "qcustomplot.h"
#include "serialportreader.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

public slots:
    void addEcgData(int data);
    void addHeartRateData(unsigned int data);

private:
    SerialPortReader *serialReader;
    double startTime;

    QCustomPlot *ecgPlot;
    QVector<double> ecgData;
    QVector<double> ecgTime;

    QCustomPlot *heartRatePlot;
    QVector<double> heartRateData;
    QVector<double> heartRateTime;

    QCPItemText *heartRateLabel;  // Add this line for the heart rate label
};

#endif // MAINWINDOW_H

import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import classification_report, confusion_matrix


def newline(p1, p2):
    """
    Draws a line between two points

    Arguments:
      p1 {list} -- coordinate of the first point
      p2 {list} -- coordinate of the second point

    Returns:
      mlines.Line2D -- the drawn line
    """
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    lines = mlines.Line2D([xmin, xmax], [ymin, ymax], linewidth=1, linestyle='--')
    ax.add_line(lines)
    return lines


def BHS_Metric_BP(x1, x2, y1, y2):
    """
    Computes the BHS Standard metric
    Arguments:
      err {array} -- array of absolute error
    Returns:
      tuple -- tuple of percentage of samples with <=5 mmHg, <=10 mmHg and <=15 mmHg error
    """

    def err_calc(G_T, pred):
        err = []

        for i in (range(len(G_T))):
            err.append(abs(G_T[i] - pred[i]))

        leq5 = 0
        leq10 = 0
        leq15 = 0

        for i in range(len(err)):

            if abs(err[i]) <= 5:
                leq5 += 1
                leq10 += 1
                leq15 += 1

            elif abs(err[i]) <= 10:
                leq10 += 1
                leq15 += 1

            elif abs(err[i]) <= 15:
                leq15 += 1

        return leq5 * 100.0 / len(err), leq10 * 100.0 / len(err), leq15 * 100.0 / len(err)

    sbp_percent = err_calc(x1, x2)
    dbp_percent = err_calc(y1, y2)
    #
    print('------------------------------------------')
    print('|        BHS-Metric-BP Prediction        |')
    print('------------------------------------------')

    print('------------------------------------------')
    print('|       | <= 5mmHg | <=10mmHg | <=15mmHg |')
    print('------------------------------------------')
    print(
        f'|  DBP  |   {round(dbp_percent[0], 2)} %   |   {round(dbp_percent[1], 2)} %   |  {round(dbp_percent[2], 2)} %    |')
    print(
        f'|  SBP  |   {round(sbp_percent[0], 2)} %   |   {round(sbp_percent[1], 2)} %   |  {round(sbp_percent[2], 2)} %    |')
    print('------------------------------------------')
    #
    err_SBP = []
    err_DBP = []

    for i in (range(len(x1))):
        err_SBP.append(abs(x1[i] - x2[i]))

    for i in (range(len(y1))):
        err_DBP.append(abs(y1[i] - y2[i]))
    #
    '''Plot figures'''
    # SBPS
    fig = plt.figure(figsize=(18, 4), dpi=120)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = ax1.twinx()
    sns.distplot(err_SBP, bins=100, kde=False, rug=False, ax=ax1)
    sns.distplot(err_SBP, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(['0%', '3.77%', '7.54%', '11.31%', '15.08%'])
    ax1.set_xlabel(r'$|$' + 'Error' + r'$|$' + ' (mmHg)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_ylabel('Percentage of Samples', fontsize=14)
    plt.title('Absolute Error in SBP Prediction', fontsize=18)
    plt.xlim(xmax=20.0, xmin=0.0)
    plt.xticks(np.arange(0, 20 + 1, 5))
    p1 = [5, 0]
    p2 = [5, 10000]
    newline(p1, p2)
    p1 = [10, 0]
    p2 = [10, 10000]
    newline(p1, p2)
    p1 = [15, 0]
    p2 = [15, 10000]
    newline(p1, p2)
    plt.tight_layout()

    # DBPS
    ax1 = plt.subplot(1, 2, 2)
    ax2 = ax1.twinx()
    sns.distplot(err_DBP, bins=100, kde=False, rug=False, ax=ax1)
    sns.distplot(err_DBP, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(['0%', '1.89%', '3.77%', '5.66%', '7.54%',
                         '9.43%', '11.31%', '13.20%', '15.08%'])
    ax1.set_xlabel(r'$|$' + 'Error' + r'$|$' + ' (mmHg)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_ylabel('Percentage of Samples', fontsize=14)
    plt.title('Absolute Error in DBP Prediction', fontsize=18)
    plt.xlim(xmax=20.0, xmin=0.0)
    plt.xticks(np.arange(0, 20 + 1, 5))
    p1 = [5, 0]
    p2 = [5, 10000]
    newline(p1, p2)
    p1 = [10, 0]
    p2 = [10, 10000]
    newline(p1, p2)
    p1 = [15, 0]
    p2 = [15, 10000]
    newline(p1, p2)
    plt.tight_layout()


def calcErrorAAMI_BP(x1, x2, y1, y2):
    def calcError(G_T, pred):
        err = []

        for i in (range(len(G_T))):
            err.append(G_T[i] - pred[i])

        return err

    sbps = calcError(x1, x2)
    dbps = calcError(y1, y2)
    #
    print('---------------------------------')
    print('| AAMI Standard - BP_Prediction |')
    print('---------------------------------')
    print('-----------------------')
    print('|     | ME | STD |     ')
    print('-----------------------')
    print('| DBP | {} |  {} |'.format(round(np.mean(dbps), 3), round(np.std(dbps), 3)))
    print('| SBP | {} |  {} |'.format(round(np.mean(sbps), 3), round(np.std(sbps), 3)))
    print('-----------------------')
    #
    err_SBP_AAMI = []
    err_DBP_AAMI = []
    #
    for i in (range(len(x1))):
        err_SBP_AAMI.append(x1[i] - x2[i])

    for i in (range(len(y1))):
        err_DBP_AAMI.append(y1[i] - y2[i])
    #
    '''Plot figures'''
    # SBPS
    fig = plt.figure(figsize=(18, 4), dpi=120)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = ax1.twinx()
    sns.distplot(err_SBP_AAMI, bins=100, kde=False, rug=False, ax=ax1)
    sns.distplot(err_SBP_AAMI, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(['0%', '1.89%', '3.77%', '5.66%', '7.54%',
                         '9.43%', '11.31%', '13.20%', '15.08%'])
    ax1.set_xlabel(r'Error' + ' (mmHg)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_ylabel('Percentage of Samples', fontsize=14)
    plt.title('Error in SBP Prediction', fontsize=18)
    plt.xlim(xmax=20.0, xmin=-20.0)
    plt.xticks(np.arange(-20, 20 + 1, 5))
    plt.tight_layout()

    # DBPS
    ax1 = plt.subplot(1, 2, 2)
    ax2 = ax1.twinx()
    sns.distplot(err_DBP_AAMI, bins=100, kde=False, rug=False, ax=ax1)
    sns.distplot(err_DBP_AAMI, bins=100, kde=False, rug=False, ax=ax2)
    ax2.set_yticklabels(['0%', '1.89%', '3.77%', '5.66%', '7.54%',
                         '9.43%', '11.31%', '13.20%', '15.08%'])
    ax1.set_xlabel(r'Error' + ' (mmHg)', fontsize=14)
    ax1.set_ylabel('Number of Samples', fontsize=14)
    ax2.set_ylabel('Percentage of Samples', fontsize=14)
    plt.title('Error in DBP Prediction', fontsize=18)
    plt.xlim(xmax=10.0, xmin=-10.0)
    plt.xticks(np.arange(-10, 10 + 1, 5))
    plt.tight_layout()


def evaluate_BP_Classification_BP(SBP_GT, DBP_GT, SBP_Pred, DBP_Pred):
    cls_gt = []
    cls_pred = []

    for i in (range(len(DBP_GT))):
        dbp_gt = DBP_GT[i]
        dbp_pred = DBP_Pred[i]

        if dbp_gt <= 80:
            cls_gt.append('Normotension')
        elif (dbp_gt > 80) and (dbp_gt <= 90):
            cls_gt.append('Pre-hypertension')
        elif dbp_gt > 90:
            cls_gt.append('Hypertension')
        else:
            print('bump')  # this will never happen, check for error

        if dbp_pred <= 80:
            cls_pred.append('Normotension')
        elif (dbp_pred > 80) and (dbp_pred <= 90):
            cls_pred.append('Pre-hypertension')
        elif dbp_pred > 90:
            cls_pred.append('Hypertension')
        else:
            print('bump')  # this will never happen, check for error
    #
    print('DBP Classification Accuracy')
    print(classification_report(cls_gt, cls_pred, digits=5))
    #
    confusion_matrix_raw_DBP = confusion_matrix(cls_gt, cls_pred, normalize=None)
    confusion_matrix_norm_DBP = confusion_matrix(cls_gt, cls_pred, normalize='true')
    shape = confusion_matrix_raw_DBP.shape
    data_DBP = np.asarray(confusion_matrix_raw_DBP, dtype=int)
    text_DBP = np.asarray(confusion_matrix_norm_DBP, dtype=float)
    annots_DBP = (np.asarray(["{0:.2f} ({1:.0f})".format(text_DBP, data_DBP) for text_DBP, data_DBP in zip(text_DBP.flatten(), data_DBP.flatten())])).reshape(shape[0],shape[1])

    # SBPS

    cls_gt = []
    cls_pred = []

    for i in (range(len(SBP_GT))):
        sbp_gt = SBP_GT[i]
        sbp_pred = SBP_Pred[i]

        if sbp_gt <= 120:
            cls_gt.append('Normotension')
        elif (sbp_gt > 120) and (sbp_gt <= 140):
            cls_gt.append('Prehypertension')
        elif sbp_gt > 140:
            cls_gt.append('Hypertension')
        else:
            print('bump')  # this will never happen, check for error

        if sbp_pred <= 120:
            cls_pred.append('Normotension')
        elif (sbp_pred > 120) and (sbp_pred <= 140):
            cls_pred.append('Prehypertension')
        elif sbp_pred > 140:
            cls_pred.append('Hypertension')
        else:
            print('bump')  # this will never happen, check for error
    #
    print('SBP Classification Accuracy')
    print(classification_report(cls_gt, cls_pred, digits=5))
    #
    confusion_matrix_raw_SBP = confusion_matrix(cls_gt, cls_pred, normalize=None)
    confusion_matrix_norm_SBP = confusion_matrix(cls_gt, cls_pred, normalize='true')
    shape = confusion_matrix_raw_SBP.shape
    data_SBP = np.asarray(confusion_matrix_raw_SBP, dtype=int)
    text_SBP = np.asarray(confusion_matrix_norm_SBP, dtype=float)
    annots_SBP = (np.asarray(["{0:.2f} ({1:.0f})".format(text_SBP, data_SBP) for text_SBP, data_SBP in zip(text_SBP.flatten(), data_SBP.flatten())])).reshape(shape[0],shape[1])
    #
    labels = ['Hypertension', 'Normotension', 'Prehypertension']
    # Plot
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix_norm_DBP, cmap='YlGnBu', annot=annots_DBP, fmt='', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix for DBP Prediction', fontsize=20)
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix_norm_SBP, cmap='YlGnBu', annot=annots_SBP, fmt='', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix for SBP Prediction', fontsize=20)
    plt.tight_layout()


def regression_plot_BP(SBP_GT, DBP_GT, SBP_Pred, DBP_Pred):
    """
      Draws the Regression Plots
    """
    sbpTrues = []
    sbpPreds = []
    dbpTrues = []
    dbpPreds = []

    for i in (range(len(SBP_GT))):
        sbpTrues.append(SBP_GT[i])
        sbpPreds.append(SBP_Pred[i])
        dbpTrues.append(DBP_GT[i])
        dbpPreds.append(DBP_Pred[i])

    '''Drawing the Regression Plots'''

    plt.figure(figsize=(18, 6), dpi=120)
    plt.subplot(1, 2, 1)
    sns.regplot(dbpTrues, dbpPreds, scatter_kws={'alpha': 0.2, 's': 1}, line_kws={'color': '#e0b0b4'})
    plt.xlabel('Target Value (mmHg)', fontsize=14)
    plt.ylabel('Estimated Value (mmHg)', fontsize=14)
    plt.title('Regression Plot for DBP Prediction', fontsize=18)
    plt.subplot(1, 2, 2)
    sns.regplot(sbpTrues, sbpPreds, scatter_kws={'alpha': 0.2, 's': 1}, line_kws={'color': '#e0b0b4'})
    plt.xlabel('Target Value (mmHg)', fontsize=14)
    plt.ylabel('Estimated Value (mmHg)', fontsize=14)
    plt.title('Regression Plot for SBP Prediction', fontsize=18)
    plt.show()

    '''
      Printing statistical analysis values like r and p value
    '''
    sbpTrues = np.array(sbpTrues).ravel()
    sbpPreds = np.array(sbpPreds).ravel()
    dbpTrues = np.array(dbpTrues).ravel()
    dbpPreds = np.array(dbpPreds).ravel()
    print('DBP')
    print(scipy.stats.linregress(dbpTrues, dbpPreds))
    print('SBP')
    print(scipy.stats.linregress(sbpTrues, sbpPreds))


def bland_altman_plot_BP(SBP_GT, DBP_GT, SBP_Pred, DBP_Pred):
    """
      Draws the Bland Altman plot
    """
    # Import Necessary Libraries
    import numpy as np
    import matplotlib.pyplot as plt
    #
    def bland_altman(data1, data2):
        """
        Computes mean +- 1.96 sd

        Arguments:
          data1 {array} -- series 1
          data2 {array} -- series 2
        """

        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        plt.scatter(mean, diff, alpha=0.1, s=4)
        plt.axhline(md, color='gray', linestyle='--')
        plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
        plt.ylim(ymin=-75, ymax=75)
        plt.xlabel('Avg. of Target and Estimated Value (mmHg)', fontsize=14)
        plt.ylabel('Error in Prediction (mmHg)', fontsize=14)
        print(md + 1.96 * sd, md - 1.96 * sd)

    sbpTrues = []
    sbpPreds = []

    dbpTrues = []
    dbpPreds = []

    for i in (range(len(SBP_GT))):
        sbpTrues.append(SBP_GT[i])
        sbpPreds.append(SBP_Pred[i])
        #
        dbpTrues.append(DBP_GT[i])
        dbpPreds.append(DBP_Pred[i])

    sbpTrues = np.array(sbpTrues).ravel()
    sbpPreds = np.array(sbpPreds).ravel()
    dbpTrues = np.array(dbpTrues).ravel()
    dbpPreds = np.array(dbpPreds).ravel()

    '''Plot the Bland Altman plot'''
    fig = plt.figure(figsize=(18, 5), dpi=120)
    plt.subplot(1, 2, 1)
    print('---------DBP---------')
    bland_altman(dbpTrues, dbpPreds)
    plt.title('Bland-Altman Plot for DBP Prediction', fontsize=18)
    plt.subplot(1, 2, 2)
    print('---------SBP---------')
    bland_altman(sbpTrues, sbpPreds)
    plt.title('Bland-Altman Plot for SBP Prediction', fontsize=18)
    plt.show()

import sys
import socket
import matplotlib.pyplot as plt
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image


def create_report(path, score, timer, rparams, tstart, history):
    """
    Create .pdf with information about experiment.
    """
    doc = SimpleDocTemplate(path+"report "+str(round(score[1], 2))+".pdf", 
    pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    Report = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
        
    string = str(rparams)
    string = string.replace("{", "")
    string = string.replace("'", "")
    string = string.replace("}", "")
    string = string.replace("\"", "")
    
    cmd = str(sys.argv)
    cmd = cmd.replace("[", "")
    cmd = cmd.replace("]", "")
    cmd = cmd.replace(",", " ")
    cmd = cmd.replace("'", "")
    
    ptext = '<font size=12> Command line input: %s </font>' % (cmd)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Parameters: %s </font>' % (string)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Score: %1.3f </font>' % (score[0])
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Accuracy: %1.3f </font>' % (score[1])
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Started at: %s </font>' % (tstart)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Time required: %s </font>' % (timer)
    Report.append(Paragraph(ptext, styles["Justify"]))
    ptext = '<font size=12> Host name: %s </font>' % (socket.gethostname())
    Report.append(Paragraph(ptext, styles["Justify"]))
    Report.append(Spacer(1, 12))
    
    draw_history(history, path)
    im = Image(path+'history.png')
    Report.append(im)
    doc.build(Report)
    
    
def draw_history(history, path):
    """
    Create plot of history and save in path.
    """
    plt.figure(1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)
    plt.subplot(211)
    
    keys=list(history.history.keys())

    plt.plot(history.history[keys[3]], color='r')
    plt.plot(history.history[keys[1]], color='g')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history[keys[2]], color='r')
    plt.plot(history.history[keys[0]], color='g')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
        
    plt.savefig(path+'history.png')
    plt.clf()

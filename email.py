import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

password = "qcof scde ezte sxwn"
me = "rohannmahajan0707@gmail.com"
you = "rohumahajan0707@gmail.com"

email_body = """<html><body><p>Hello World</p></body></html>"""

message = MIMEMultipart('alternative', None, [MIMEText(email_body, 'html')])

message['Subject'] = 'Test email send'
message['From'] = me
message['To'] = you

try:
  server = smtplib.SMTP('smtp.gmail.com:587')
  server.ehlo()
  server.starttls()
  server.login(me, password)
  server.sendmail(me, you, message.as_string())
  server.quit()
  print(f'Email sent: (email_body)')
except Exception as e:
  print(f'Error in sending email: {e}')

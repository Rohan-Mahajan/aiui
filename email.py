I want you to make the following changes in the code - 
1. make a file where you can do all the logging part, wherever you are using this logging, it should be done in a single file, that too in a proper format.
2. In the test_cases CSV, there is no need to classify the test cases in categories like positive or negative, you just need to make sure that you are fetching the correct test cases from CSV. Eliminate the calssify_test_case function, and code related to it.
3. The Agent should first look in the CSV file and fetch the test cases needed to validate the solution, and then if required, generate the test cases so that the solution can be validated end-to-end, properly.
4. When the defects are not found in the defects.csv, agent should generate the solution, and test cases, and those test cases should also validate the solution end-to-end, instead of just giving 2 positive and 2 negative solutions. 
5. There should be a function that can send the output, like the defect, the solution and the test cases to the user, via an email. For now, Agent should have my email - rohannmahajan@gmail.com as the sender's email, and ronny@gmail.com as the receiver's mail. And this mail must be sent every time. use this template to send the email - 
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

password = "qcof scde ezte sxwn"
me = "rohannmahajan0707@gmail.com"
you = "rohumahajan0707@gmail.com"

email_body = f"""<html><body><p>{final_solution}</p></body></html>"""

message = MIMEMultipart('alternative', None, [MIMEText(email_body, 'html')])

message['Subject'] = 'Defect RCA'
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

please do these following changes for me, i request you.

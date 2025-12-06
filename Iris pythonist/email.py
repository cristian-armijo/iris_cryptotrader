import smtplib 
from email.message import EmailMessage 

email_subject = "Email test from Python" 
sender_email_address = "carmijo@favoritafc.com" 
receiver_email_address = "arvactur3@gmail.com" 
email_smtp = "smtp.office365.com" 
email_password = "Arvactur.2" 

# Create an email message object 
message = EmailMessage() 

# Configure email headers 
message['Subject'] = email_subject 
message['From'] = sender_email_address 
message['To'] = receiver_email_address 

# Set email body text 
message.set_content("script usando python") 

# Set smtp server and port 
server = smtplib.SMTP(email_smtp, '587') 

# Identify this client to the SMTP server 
server.ehlo() 

# Secure the SMTP connection 
server.starttls() 

# Login to email account 
server.login(sender_email_address, email_password) 

# Send email 
server.send_message(message) 

# Close connection to server 
server.quit()


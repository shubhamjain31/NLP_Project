from django.shortcuts import render
from django.http import HttpResponse
from App import SentimentAnalyzer, SpamAnalyzer, TextSummarizer, Resume

# Create your views here.

def index(request):
	return render(request,'index.html')

def sentiment(request):
	if request.method == 'POST':
		value = request.POST.get('message')
		message = SentimentAnalyzer.sentiment_analyse(value)
		return render(request,'Sentiment.html',{'prediction':message})
	return render(request,'Sentiment.html')

def spam(request):
	if request.method == 'POST':
		value = request.POST.get('message')
		message = SpamAnalyzer.spam_analyse(value)		
		return render(request,'Spam.html',{'prediction':message})
	return render(request,'Spam.html')

def summarize(request):
	if request.method == 'POST':
		value = request.POST.get('message')
		message = TextSummarizer.text_analyse(value)
		print(message)		
		return render(request,'Summarize.html',{'original':value,'prediction':message})
	return render(request,'Summarize.html')

def resumeparser(request):
	if request.method == 'POST':
		value = request.FILES['filename']
		file_path = 'App/Files/'+str(value)

		# calling above function and extracting text
		text = ''
		for page in Resume.extract_text_from_pdf(file_path):
		    text += ' ' + page
		message = Resume.extract_skills(text)
		return render(request,'Resume.html',{'message':message})
	return render(request,'Resume.html')
from googletrans import Translator
sentence = input("Enter a sentence: ")
translator = Translator()
result = translator.translate(sentence,src='en', dest='hi')
print(result.text)

# To get other language code fro src or dest you can go to
# https://www.labnol.org/code/19899-google-translate-languages
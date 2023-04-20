# An Evaluation of the Accuracy of Google Cloud Translation API
## Introduction
### Research Question 
- What are some factors that impact the quality of translation of Google Cloud Translation API?

### Explanatory Variables
1. Target Language 
2. Type of Texts 
3. Number of Times The Text is Translated

### Response Variable
1. Accuracy of Translation (measured by cosine similarity score between the original text and the translated text)

### Goal of This Project
- The main goal of this project is to test the quality of the Google Cloud Translation API. The quality of translation is measured by the text similarity score between original text and the final text after being translated back and forth. 

### Hypothesis
- Our hypothesis is that the API would have varied performance when translating different languages and content types. Our other hypothesis is that text similarity scores will go down as we increase the number of translations since the original meaning of certain texts will be lost through translation.

### Text Input
|Content Type|Article|
|------------|-------------|
|News Article|[Trump Asks to Delay Sexual Assault Trial Following Historic Indictment](https://www.cnn.com/2023/04/12/politics/e-jean-carroll-trump-lawyers-trial/index.html)|
|Scientific Article|[How Hard Wired is Human Behavior](https://hbr.org/1998/07/how-hardwired-is-human-behavior)|
|Philosophical Essay|[Introduction to Critique of Pure Reason](https://www.marxists.org/reference/subject/ethics/kant/reason/ch01.htm)|
|Speech|[Barack Obama's 2008 New Hampshire Primary Speech](https://gist.github.com/mcdickenson/25479c8571b8f86f3a21c8d579102f93)|

## Architecture Diagram
User would first upload or manually type down the input text into the google colab notebook. In the notebook, the user would also define parameters including the target language and number of rounds that the input text will be translated back and forth between the original langauge and the target language. A Google Cloud Translation API would then be initialized and used to translate text and receive translation results. Finally, the results are organized into dataframes and analyzed in the google colab notebook. 
<img width="924" alt="Screen Shot 2023-04-20 at 9 33 24 AM" src="https://user-images.githubusercontent.com/120674894/233382738-4723e049-c01c-491f-a5a6-176f2752c367.png">

## Wordflow Diagram
<img width="561" alt="Screen Shot 2023-04-13 at 2 31 36 PM" src="https://user-images.githubusercontent.com/120674894/231851426-153404a2-c977-40e9-9465-d282486b0493.png">

## Using the API in Colab Notebook
- First, manually type down a piece of text and define parameters
<img width="1368" alt="Screen Shot 2023-04-18 at 9 21 25 AM" src="https://user-images.githubusercontent.com/120674894/232790711-48dba992-092d-465c-9c20-fc042dbd73eb.png">

- Based on the parameters defined, the translated text will be printed along with other requested outputs
<img width="444" alt="Screen Shot 2023-04-18 at 9 21 40 AM" src="https://user-images.githubusercontent.com/120674894/232790802-6dedf8ba-3759-413e-ae2a-7a19dfa93a03.png">


## Code Explanation
### Upload Credentials
1. Enable Cloud Translation API in your project
2. Go to API & Services -> Credentials -> Create Credentails 
3. Create a service account, grant the account Cloud Translation API Editor role, and use the service account to generate API credentials in json format
4. Name the API Credentials "api_credentials.json", upload it using the following code and store it as an environmental variable named "GOOGLE_APPLICATION_CREDENTAILS" using the following code
```python
files.upload()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "api_credentials.json"
```

### Helper Functions
- Define a function that converts two texts into all lowercase, remove punctuations, and computes the cosine similarity score of two texts
```python
def text_similarity(text1, text2):
  # remove punctuations and make text all lowercase
  text1 = re.sub('[^\w\s]', '', text1).lower()
  text2 = re.sub('[^\w\s]', '', text2).lower()

  vectorizer = TfidfVectorizer()
  vectors = vectorizer.fit_transform([text1, text2])
  similarity = cosine_similarity(vectors)
  return similarity[1,0]
```

- Define a function that calls google cloud translation api, translates a text between the original language and the target language for a specified number of rounds, and returns the translated text
```python
# import library and upload api credentials
from google.cloud import translate_v2
files.upload()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "api_credentials.json"

def repeat_translate(text, original_language, target_language, num_rounds):
  translator = translate_v2.Client()
  if (original_language is None):
    original_language = translator.detect_language(text)["language"]

  intermediate_text_original = text
  intermediate_text_target = None
  for round in range(0, num_rounds):
    intermediate_text_target = translator.translate(intermediate_text_original, 
                                             target_language = target_language,
                                             model = "nmt")["translatedText"]
    intermediate_text_original = translator.translate(intermediate_text_target, 
                                             target_language = original_language,
                                             model = "nmt")["translatedText"]
  return [intermediate_text_original, intermediate_text_target]
```

### Fetching Data from API
- Each forloop iteration translates a text for a specified number of rounds, compares the original with the the translated text to compute a similarity score, then add the data to the result dataframe
```python
# define variables
filenames = ['philosophical_essay.txt', 'news_article.txt', 'scientific_article.txt', 'speech.txt'] # to be uploaded
var1_text = [open(filename).read().replace('\n', '') for filename in filenames] # to be uploaded
var2_target_language = ["French", "German", "Mandarin", "Russian", "Latin", "Korean", "Japanese", "Arabic"]
var3_num_translations = [1, 2, 5, 10]

# define an empty dataframe to store data
result = pd.DataFrame(columns = ["Original_Language", "Target_Language", "Content_Type", "Num_Rounds", "Similarity_Score"])
original_language = "English"

# translate the input texts for a specified number of rounds
for text_idx in range(0, len(var1_text)):
  text_name = filenames[text_idx].split('.')[0]
  original_text = var1_text[text_idx]
  for target_language in var2_target_language:
    for num_rounds in var3_num_translations:
      translated_text = repeat_translate(original_text, language_map[original_language], language_map[target_language], num_rounds)[0]
      similarity_score = text_similarity(original_text, translated_text)

      # append new row to dataframe
      new_row = [original_language, target_language, text_name, num_rounds, similarity_score]
      result.loc[len(result)] = new_row
```
- Result dataframe overview

 <img width="702" alt="Screen Shot 2023-04-16 at 2 24 35 PM" src="https://user-images.githubusercontent.com/120674894/232333776-1863793a-1f40-481d-900a-8acf0acd0b18.png">
 
### Data Visualizations
- To visualize the relationship between individual explanatory variables and similarity score, data is grouped by the explanatory variable and then averaged by the response variable. 
```python
# reorganized data
result_content = result.groupby("Content_Type").agg(Average_Similarity_Score = ("Similarity_Score", "mean")).sort_values("Average_Similarity_Score").reset_index()

# plot data
result_content.plot("Content_Type", "Average_Similarity_Score", kind = "barh")
display(result_content)
plt.legend([])
plt.xlabel("Type of Text")
plt.ylabel("Target Language")
plt.title("Average Similarity Score for Different Content Types")
plt.xlim(0.75, 1)

for i in range(0, len(result_content)):
score = result_content.loc[i, "Average_Similarity_Score"]
plt.text(score, i, str(round(score, 3)), fontweight = "bold")
```
- The above code generates the following table and plot

<img width="373" alt="Screen Shot 2023-04-16 at 2 33 39 PM" src="https://user-images.githubusercontent.com/120674894/232334729-8aa4ffce-9b6d-4c29-95d4-79c70f04052b.png">
<img width="709" alt="Screen Shot 2023-04-16 at 2 33 58 PM" src="https://user-images.githubusercontent.com/120674894/232334735-0a46fd56-7377-4f49-94de-04b77ae9965b.png">


- To visualize how two variables work in tandem to impact similarity score, data is reorganized into tables where the row index represents one variable and the column index represents another variable
```python
result_type_numrounds = pd.pivot_table(result, index = "Content_Type", columns = "Num_Rounds", values = "Similarity_Score").T
display(result_type_numrounds)
for column in result_type_numrounds.columns:
  result_type_numrounds[column].plot()
plt.xlabel("Number of Rounds of Translation")
plt.ylabel("Similarity Score")
plt.title("Similarity Score for Different Content Type")
plt.legend(bbox_to_anchor = (1.05, 1))
plt.show()
```

- The above code generates the following table and plot
<img width="641" alt="Screen Shot 2023-04-16 at 2 42 13 PM" src="https://user-images.githubusercontent.com/120674894/232334685-083d4188-6455-48cf-97dd-80df5d6eebe4.png">
<img width="794" alt="Screen Shot 2023-04-16 at 2 42 37 PM" src="https://user-images.githubusercontent.com/120674894/232334689-af5ea0f2-5e53-4541-8be7-d8757cbbc0da.png">

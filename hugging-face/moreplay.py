from transformers import AutoModel, AutoTokenizer, pipeline

# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# sum = summarizer("""Gaga was best known in the 2010s for pop hits like
# “Poker Face” and avant-garde experimentation on albums like “Artpop,” and
# Bennett, a singer who mostly stuck to standards, was in his 80s when the
# pair met. And yet Bennett and Gaga became fast friends and close
# collaborators, which they remained until Bennett’s death at 96 on Friday.
# They recorded two albums together, 2014’s “Cheek to Cheek” and 2021’s
# “Love for Sale,” which both won Grammys for best traditional pop vocal
# album.""", min_length=20, max_length=50)

# print(sum[0]['summary_text'])


# classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
# lbl = classifier("This restaurant was average")

# print(lbl)

generator = pipeline(model="gpt2")
ouput = generator("Zac, Conor, and Rory are very good brothers, but sometimes", do_sample=True, top_p=0.95, num_return_sequences=4, max_new_tokens=50, return_full_text=False)

for item in ouput:
    print(">", item['generated_text'])

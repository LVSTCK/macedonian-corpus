from datatrove.utils.lid import FT176LID, GlotLID, FastTextLID
from datatrove.data import Document

model = FT176LID(languages = ["mk"])

# define a document 
example = Document(
                    text="Ова е пример текст на македонски јазик. How about this? Is this full mkd? ",
                    id=0,
                    metadata={},
                    )

print(model.predict(example))
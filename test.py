from IPython.display import Video, display, Javascript

def create_modal(title, body):
    title = title
    body = body
    
    display(Javascript("""
    require(
        ["base/js/dialog"], 
        function(dialog) {
            dialog.modal({
                title: {title},
                body: {body},
                buttons: {
                    'Okay': {}
                }
            });
        })
    """))

title = "Feature Extraction"
modal_text = "Feature Extraction Successful! RGB .npy files saved to i3d-feature-extraction/output/RGB"

create_modal(title, modal_text)  

# if "Saving features" in output:
  
# else:
    
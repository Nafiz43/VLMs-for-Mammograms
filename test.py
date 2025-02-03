for report in range(0, 509):
        report_id = str(report).zfill(3) 
        # query = 'image ID: ' + report_id

        query =  'Process the report having the image ID: IMG'+  report_id+'.png'
        print(query)
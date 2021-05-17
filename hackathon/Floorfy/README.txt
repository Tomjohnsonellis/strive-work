The dataset is divided in 2 parts:

1) IMAGES OF TOURS [31.6 GB]
Filename: "tours_divided/"

Composed by 409 tours of different agencies separated in a directory each, whose name corresponds to the tour id ("property_id" attribute
in the user visits JSON). In total, there are 10000 4K RGB equirectangular images.

We have divided the dataset into 10 .zip with approximately 1000 images inside each one. Some statistics about the diferent .zip is shown
below:

			  |  SIZE [GB] |  NUM. TOURS |  NUM. IMAGES  |  MIN. IMAGES IN A TOUR | MAX. IMAGES IN A TOUR | MEAN NUM. IMAGES PER TOUR 
--------------------------------------------------------------------------------------------------------------------------------------
tours_1.zip   |    3.63	   |	  48     |     1011		 |      	5 			  | 		  76 		  | 		21.06
--------------------------------------------------------------------------------------------------------------------------------------
tours_2.zip   |	   3.29    |      53     |     1004	     | 		    7 			  |			  78 		  | 		18.94
--------------------------------------------------------------------------------------------------------------------------------------
tours_3.zip   |	   2.99	   |      37     |     1002		 | 			8			  |			  88 		  | 		27.08
--------------------------------------------------------------------------------------------------------------------------------------
tours_4.zip   |    3.10	   |      24     |     1030      | 			15			  | 		  131 		  | 		42.90
--------------------------------------------------------------------------------------------------------------------------------------
tours_5.zip   |    2.98	   |      28     |     1016      | 			12			  | 		  65		  | 		36.28
--------------------------------------------------------------------------------------------------------------------------------------
tours_6.zip   |    3.20    |      30     |     996 		 | 			6 			  | 		  124		  | 		33.20
--------------------------------------------------------------------------------------------------------------------------------------
tours_7.zip   |    3.17    |	  77     |     1006	     | 			1 			  | 		  45		  |			13.06
--------------------------------------------------------------------------------------------------------------------------------------
tours_8.zip   |    2.99    |      40     |     996		 | 			10			  | 		  88		  |		 	24.90
--------------------------------------------------------------------------------------------------------------------------------------
tours_9.zip   |    3.22    |      38     |     1030		 |			13			  | 		  47		  | 		27.11
--------------------------------------------------------------------------------------------------------------------------------------
tours_10.zip  |    2.94    |      34	 |     909 		 | 			8			  | 		  55		  | 		26.73
======================================================================================================================================
	TOTAL     |    31.51   |      409    |     10000     |          1 			  | 		  131		  | 		27.13



2) ANALYTICS OF USER VISITS [35.3 MB]
Filename: "user_visit_analytics.json"

We provide 10000 JSON objects with the user's visits information. Each line of the file represents one JSON object. Some of them
are from the 409 tours provided.

The important attributes of the JSON are shown below:

	- id: Integer. Id of the analytic.

	- route: JSONArray. It contains all the information of the client's path along the tour. Every JSONObject has 5 attributes:

		· id: String.
			  The equirectangular image id.

		· isAutomatic: String ["true" | "false"].
			  If "true", the scene transitions automatically, therefore the user has not changed the scene using the menu nor the hotspots.

		· sceneVRmode: String ["true" | "false"].
		      If "true", the user is using VR.

		· scenetransmode: String ["begin", "menu", "hotspot"].
		      How the scene has been changed. If "begin", the scene is the first one, if "menu" the transition has been done using
		      the menu, if "hotspot" the user has clicked on the hotspot circles.

		· scenetime: String.
		      The total time the user has been in a particular scene.


	- property_id: Integer. With this id, you can visit the tour at "https://floorfy.com/tour/{property_id}". Also, you can check
		if it belongs to one of the 409 tours provided.

	- total_time: Integer. The total time the user has been on the tour.

	- agency_id: Integer. The id of the agency.

	- date. JSONObject. It contains the attribute $date (Integer) which has the timestamp of the visit.

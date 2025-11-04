from landlab import Component
import numpy as np
from landlab import RasterModelGrid, imshow_grid

class Erodible_Grid(Component):
    _name = "ErodibleGrid"
    _unit_agnostic = True
    
    def __init__(
            self, 
            nrows,
            ncols,
            spacing,
            full_tire,
            long_slope
    ):
        """Initialize Erodible Grid Component"""
        super().__init__(None)

        mg = RasterModelGrid((nrows,ncols),spacing)
        z = mg.add_zeros('topographic__elevation', at='node') #create the topographic__elevation field
        road_flag = mg.add_zeros('flag', at='node') #create a road_flag field for determining whether a 
                                                        #node is part of the road or the ditch line
        n = mg.add_zeros('roughness', at='node') #create roughness field
            
        mg.set_fixed_value_boundaries_at_grid_edges(True, True, True, True)     
        mg.set_closed_boundaries_at_grid_edges(False, False, False, False) 

        if full_tire == False: #When node spacing is half-tire-width
            road_peak = 40 #peak crowning height occurs at this x-location
            up = 0.0067 #rise of slope from ditchline to crown
            down = 0.0067 #rise of slope from crown to fillslope
            
            for g in range(nrows): #loop through road length
                elev = 0 #initialize elevation placeholder
                flag = False #initialize road_flag placeholder
                roughness = 0.1 #initialize roughness placeholder   

                for h in range(ncols): #loop through road width
                    if h == 0 or h == 8:
                        elev = 0
                        flag = False
                        roughness = 0.1
                    elif h == 1 or h == 7:
                        elev = -0.109
                        flag = False
                        roughness = 0.1
                    elif h == 2 or h == 6:
                        elev = -0.1875
                        flag = False
                        roughness = 0.1
                    elif h == 3 or h == 5:
                        elev = -0.2344
                        flag = False
                        roughness = 0.1
                    elif h == 4:
                        elev = -0.25
                        flag = False
                        roughness = 0.1
                    elif h <= road_peak and h > 8: #update latitudinal slopes based on location related to road_peak
                        elev += up
                        flag = True
                        roughness = 0.05
                    else:
                        elev -= down
                        flag = True
                        roughness = 0.05

                    z[g*ncols + h] = elev #update elevation based on x & y locations
                    road_flag[g*ncols+h] = flag #update road_flag based on x & y locations
                    n[g*ncols + h] = roughness #update roughness values based on x & y locations
        elif full_tire == True: #When node spacing is full-tire-width
            road_peak = 20 #peak crowning height occurs at this x-location
            up = 0.0134 #rise of slope from ditchline to crown
            down = 0.0134 #rise of slope from crown to fillslope
            
            for g in range(nrows): #loop through road length
                elev = 0 #initialize elevation placeholder
                flag = False #initialize road_flag placeholder
                roughness = 0.1 #initialize roughness placeholder

                for h in range(ncols): #loop through road width
                    if h == 0 or h == 4:
                        elev = 0
                        flag = False
                        roughness = 0.1
                    elif h == 1 or h == 3:
                        elev = -0.1875
                        flag = False
                        roughness = 0.1
                    elif h == 2:
                        elev = -0.25
                        flag = False
                        roughness = 0.1
                    elif h <= road_peak and h > 4: #update latitudinal slopes based on location related to road_peak
                        elev += up
                        flag = True
                        roughness = 0.05
                    else:
                        elev -= down
                        flag = True
                        roughness = 0.05

                    z[g*ncols + h] = elev #update elevation based on x & y locations
                    road_flag[g*ncols+h] = flag #update road_flag based on x & y locations
                    n[g*ncols + h] = roughness #update roughness values based on x & y locations
        
        z += mg.node_y*long_slope #add longitudinal slope to road segment
        road_flag = road_flag.astype(bool) #Make sure road_flag is a boolean array
        self.mg = mg
        self.z = z
        self.road_flag = road_flag
        self.n = n

    def __iter__(self):
        """Allows unpacking the component directly."""
        return iter((self.mg, self.z, self.road_flag, self.n))

    def __call__(self):
        """Allows returning the four objects by calling the instance."""
        return self.mg, self.z, self.road_flag, self.n

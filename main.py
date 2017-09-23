import sys
import os
import datetime

from PyQt5 import QtCore, QtGui, QtWidgets, uic

from tinyekf import EKF
from scipy import stats
import numpy as np
import cv2

from collections import defaultdict

qtCreatorFile = "ui/mousetrack.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

# ----- CLASS FOR KALMAN FILTER ----- #

class TrackerEKF(EKF):
	'''
	An EKF for mouse tracking
	'''

	def __init__(self):

		# Four states, two measurements (X,Y)
		EKF.__init__(self, 2, 2)

	def f(self, x):

		# State-transition function is identity
		return np.copy(x)

	def getF(self, x):

		# So state-transition Jacobian is identity matrix
		return np.eye(2)

	def h(self, x):

		# Observation function is identity
		return x

	def getH(self, x):

		# So observation Jacobian is identity matrix
		return np.eye(2)

# ----- CLASSES FOR DETECTED OBJECTS ----- #

class Hole:
	def __init__(self, center, radius, init_index):
		self.center = center
		self.radius = radius
		self.init_index = init_index

		self.number = 0
		self.isTarget = False

		self.isActive = False

		self.numFrames = 0
		self.numFramesPrimary = 0
		self.numFramesSecondary = 0

		self.numTimes = 0
		self.numTimesPrimary = 0
		self.numTimesSecondary = 0

	def setActive(self, TARGET_FLAG):
		if not(self.isActive):
			# when hole switches from inactive to active, increment the number of time it has been visted
			self.numTimes = self.numTimes + 1

			if not(self.isTarget):
				# if the hole isn't a target hole, then this visit is an error
				if TARGET_FLAG:
					# if the target has been visited already, secondary error
					self.numTimesSecondary = self.numTimesSecondary + 1
				else:
					# else primary error
					self.numTimesPrimary = self.numTimesPrimary + 1

			self.isActive = True

		# always increment number of active frames
		self.numFrames = self.numFrames + 1

		if not(self.isTarget):
			# if the hole isn't a target hole, increment the number of error frames
			if TARGET_FLAG:
				self.numFramesSecondary = self.numFramesSecondary + 1
			else:
				self.numFramesPrimary = self.numFramesPrimary + 1

class Table:
	def __init__(self, center, radius):
		self.center = center
		self.radius = radius

class Mouse:
	def __init__(self):
		
		self.id = ''

		# measured positions
		self.center = []
		self.head = []
		self.tail = []

		# estimated positions
		self.center_k = []
		self.head_k = []
		self.tail_k = []

		# kalman filters for positions
		self.kalfilt_center = TrackerEKF()
		self.kalfilt_head = TrackerEKF()
		self.kalfilt_tail = TrackerEKF()

		# tracking counters
		self.center_track_reset = 0
		self.head_track_reset = 0
		self.tail_track_reset = 0

	def getNearestHole(self, holes, point):

		if point == 'head':
			# then track the tail, because my code is fucked up and it tracks the tail as the head most of the time
			tracking_point = self.tail_k[-1]
		elif point == 'tail':
			# then track the head, because, once again, my code is fucked up and it tracks the head as the tail most of the time
			tracking_point = self.head_k[-1]
		else:
			# track the center in all other cases
			tracking_point = self.center_k[-1]

		distances = []
		for hole in holes:
			distance = np.sqrt((hole.center[0]-tracking_point[0])**2 + (hole.center[1]-tracking_point[1])**2)
			distances.append(distance)

		sorted_by_distance = sorted(zip(distances, holes), key=lambda x: x[0], reverse=False)
		return sorted_by_distance[0][1], np.amin(distances)

	def getPathLength(self, point, pixel_to_cm):
		path_length = 0
		if point == 'head':
			# then track the tail, because my code is fucked up and it tracks the tail as the head most of the time
			path = self.tail
		elif point == 'tail':
			# then track the head, because, once again, my code is fucked up and it tracks the head as the tail most of the time
			path = self.head
		else:
			# track the center in all other cases
			path = self.center

		for i in range(1,len(path)):
			path_length = path_length + np.sqrt((path[i][0]-path[i-1][0])**2 + (path[i][1]-path[i-1][1])**2)*pixel_to_cm

		# print np.sqrt((path[-1][0]-path[0][0])**2 + (path[-1][1]-path[0][1])**2)*pixel_to_cm, pixel_to_cm
		# print path
		# print ''

		return path_length

	def getPath(self, point):
		if point == 'head':
			# then track the tail, because my code is fucked up and it tracks the tail as the head most of the time
			path = self.tail
		elif point == 'tail':
			# then track the head, because, once again, my code is fucked up and it tracks the head as the tail most of the time
			path = self.head
		else:
			# track the center in all other cases
			path = self.center

		return path

	def getProxToTarget(self, point, target, pixel_to_cm):
		prox_to_target = []
		if point == 'head':
			# then track the tail, because my code is fucked up and it tracks the tail as the head most of the time
			path = self.tail
		elif point == 'tail':
			# then track the head, because, once again, my code is fucked up and it tracks the head as the tail most of the time
			path = self.head
		else:
			# track the center in all other cases
			path = self.center

		for i in range(1,len(path)):
			distance = np.sqrt((target.center[0]-path[i][0])**2 + (target.center[1]-path[i][1])**2)*pixel_to_cm
			prox_to_target.append(distance)

		return prox_to_target


# ----- MAIN APPLICATION CLASS ----- #

class MyApp(QtWidgets.QWidget, Ui_MainWindow):
	
	# ----- METHODS FOR SETTING UP THE GUI ----- #

	def __init__(self):

		QtWidgets.QWidget.__init__(self)
		Ui_MainWindow.__init__(self)
		self.setupUi(self)

		self.setBindings('Video Analysis')
		self.setBindings('Save Session')
		self.setBindings('Load Session')
		self.setBindings('Summarize Results')

	def setBindings(self, tab):

		# //
		#    This method sets the method bindings for all the buttons in the GUI.
		# //

		if tab == 'Video Analysis':
			
			self.setWorkingDirectoryButton.clicked.connect(self.setWorkingDirectory)

			self.loadVideoButton.clicked.connect(self.loadVideo)
			self.loadVideoButton.setEnabled(False)

			self.trialInfoButton.clicked.connect(self.confirmTrialInfo)
			self.trialInfoButton.setEnabled(False)

			self.checkTargetBoxButton.clicked.connect(self.selectTarget)
			self.checkTargetBoxButton.setEnabled(False)

			self.specifyTargetBoxButton.clicked.connect(self.specifyTargetBox)
			self.specifyTargetBoxButton.setEnabled(False)

			self.confirmAnalysisParametersButton.clicked.connect(self.confirmAnalysisParameters)
			self.confirmAnalysisParametersButton.setEnabled(False)

			self.headTailYes.clicked.connect(self.setUserSwapNo)
			self.headTailYes.setEnabled(False)

			self.headTailNo.clicked.connect(self.setUserSwapYes)
			self.headTailNo.setEnabled(False)

			self.runAnalysisButton.clicked.connect(self.runSingleVideoAnalysis)
			self.runAnalysisButton.setEnabled(False)

		elif tab == 'Save Session':
			
			self.saveSessionButton.clicked.connect(self.saveSession)
			self.saveSessionButton.setEnabled(False)

		elif tab == 'Load Session':
			
			self.loadSessionButton.clicked.connect(self.loadSessions)

			self.confirmAnalysisParametersButton_2.clicked.connect(self.confirmAnalysisParameters_2)
			self.confirmAnalysisParametersButton_2.setEnabled(False)
			
			self.runAnalysesButton.clicked.connect(self.runMultipleVideoAnalysis)
			self.runAnalysesButton.setEnabled(False)

		elif tab == 'Summarize Results':

			self.testingInformationButton.clicked.connect(self.confirmTestingInfo)

			self.summarizeResultsButton.clicked.connect(self.summarizeResults)
			self.summarizeResultsButton.setEnabled(False)

	# ----- METHODS FOR VIDEO ANALYSIS ----- #
	
	def initializeVariables(self):

		# //
		#    This method initializes variables that will be used during the video analysis.
		# //

		self.frame_count = 0

		self.MONTH = 0
		self.DAY = 0
		self.TRIAL = 0
		self.MOUSE_NAME = ''
		self.TARGET = 0

		self.cap = cv2.VideoCapture(str(self.vid_filename))
		self.fps = self.cap.get(cv2.CAP_PROP_FPS)
		self.spf = (1/self.fps) * 2 
		# 					 ^--- multiply by two to get correct timing 
		#						 ... a what the fuck black magic thing with opencv 
		#						 ... (read() function seems to read every other frame)

		self.mouse = Mouse()

		self.TARGET_FLAG = False
		self.target_hole_activity = 0
		self.target_hole_end = False

		self.primary_path_length = 0
		self.primary_latency = 0

	def confirmTrialInfo(self):

		# //
		#    This methods loads in the trial information and sets the mouse id.
		#    It also creates the output directory, which is based on the trial information
		# //

		self.MONTH = int(self.timePointLineEdit.text())
		self.DAY = int(self.dayLineEdit.text())
		self.TRIAL = int(self.trialLineEdit.text())
		self.MOUSE_NAME = str(self.mouseIDLineEdit.text())

		self.mouse.id = self.MOUSE_NAME + '_m' + str(self.MONTH) + 'd' + str(self.DAY) + 't' + str(self.TRIAL)
		self.consoleOutputText.append('Mouse ID: ' + self.mouse.id)

		output_directory = 'results/' + self.MOUSE_NAME
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)
		self.consoleOutputText.append('Output Directory: ' + output_directory)

		self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.output_filename = output_directory + '/' + self.mouse.id + '_tracked.avi'
		self.consoleOutputText.append('Output File: ' + self.output_filename + '\n')

		self.confirmAnalysisParametersButton.setEnabled(True)

	def confirmTestingInfo(self):
		self.summarize_ids = str(self.mouseIDsLineEdit.text()).replace(' ','').split(',')
		self.summarize_timepoint = str(self.timePointLineEdit_summarize.text())
		self.summarize_days = str(self.dayLineEdit_summarize.text()).replace(' ','').split(',')
		self.summarize_trials = str(self.numTrialsLineEdit.text()).replace(' ','').split(',')

		self.consoleOutputText.append('Summarize Results From:')
		self.consoleOutputText.append(str(self.summarize_ids))
		self.consoleOutputText.append('Days: ' + str(self.summarize_days))
		self.consoleOutputText.append('Number of Trials Performed: ' + str(self.summarize_trials))

		self.summarizeResultsButton.setEnabled(True)


	def confirmAnalysisParameters(self):

		# //
		#    This method loads in the video analysis parameters from the 'Video Analysis' tab of the GUI.
		# //
		
		self.BLUR_KERNEL_WIDTH = int(self.BLUR_KERNEL_WIDTH_INPUT.text())
		self.BLUR_KERNEL_HEIGHT = int(self.BLUR_KERNEL_HEIGHT_INPUT.text())

		self.HOLE_MASK_FACTOR = int(self.HOLE_MASK_FACTOR_INPUT.text())
		self.PREV_DIST_LIM_1 = int(self.PREV_DIST_LIM_1_INPUT.text())
		self.PREV_DIST_LIM_2 = int(self.PREV_DIST_LIM_2_INPUT.text())
		self.RESET_THRESH = int(self.RESET_THRESH_INPUT.text())

		self.MAX_CORNERS = int(self.MAX_CORNERS_INPUT.text())
		self.QUALITY_LEVEL = float(self.QUALITY_LEVEL_INPUT.text())
		self.MIN_DISTANCE = int(self.MIN_DISTANCE_INPUT.text())
		self.BLOCK_SIZE = int(self.BLOCK_SIZE_INPUT.text())
		self.FEATURE_PARAMS = dict( maxCorners = self.MAX_CORNERS,
									qualityLevel = self.QUALITY_LEVEL,
									minDistance = self.MIN_DISTANCE,
									blockSize = self.BLOCK_SIZE)

		self.RANGE_FOR_ACTIVATION = float(self.RANGE_FOR_ACTIVATION_INPUT.text())
		self.TARGET_HOLE_ACTIVITY = int(self.TARGET_HOLE_ACTIVITY_INPUT.text())

		self.KALMAN_UPDATE_PARAM = int(self.KALMAN_UPDATE_PARAM_INPUT.text())

		self.MIN_RADIUS_HOLE = int(self.MIN_RADIUS_HOLE_INPUT.text())
		self.MAX_RADIUS_HOLE = int(self.MAX_RADIUS_HOLE_INPUT.text())
		self.HOLE_DIAMETER = int(self.HOLE_DIAMETER_INPUT.text())

		self.DILATE_KERNEL_WIDTH = int(self.DILATE_KERNEL_WIDTH_INPUT.text())
		self.DILATE_KERNEL_HEIGHT = int(self.DILATE_KERNEL_HEIGHT_INPUT.text())

		self.FRAME_LIM = -1

		self.checkTargetBoxButton.setEnabled(True)

		self.consoleOutputText.append('Video analysis parameters confirmed.\n')

	def confirmAnalysisParameters_2(self):
		# //
		#    This method loads in the video analysis parameters from the 'Load Sessions' tab of the GUI.
		# //
		
		self.BLUR_KERNEL_WIDTH = int(self.BLUR_KERNEL_WIDTH_INPUT_2.text())
		self.BLUR_KERNEL_HEIGHT = int(self.BLUR_KERNEL_HEIGHT_INPUT_2.text())

		self.HOLE_MASK_FACTOR = int(self.HOLE_MASK_FACTOR_INPUT_2.text())
		self.PREV_DIST_LIM_1 = int(self.PREV_DIST_LIM_1_INPUT_2.text())
		self.PREV_DIST_LIM_2 = int(self.PREV_DIST_LIM_2_INPUT_2.text())
		self.RESET_THRESH = int(self.RESET_THRESH_INPUT_2.text())

		self.MAX_CORNERS = int(self.MAX_CORNERS_INPUT_2.text())
		self.QUALITY_LEVEL = float(self.QUALITY_LEVEL_INPUT_2.text())
		self.MIN_DISTANCE = int(self.MIN_DISTANCE_INPUT_2.text())
		self.BLOCK_SIZE = int(self.BLOCK_SIZE_INPUT_2.text())
		self.FEATURE_PARAMS = dict( maxCorners = self.MAX_CORNERS,
									qualityLevel = self.QUALITY_LEVEL,
									minDistance = self.MIN_DISTANCE,
									blockSize = self.BLOCK_SIZE)

		self.RANGE_FOR_ACTIVATION = float(self.RANGE_FOR_ACTIVATION_INPUT_2.text())
		self.TARGET_HOLE_ACTIVITY = -1

		self.KALMAN_UPDATE_PARAM = int(self.KALMAN_UPDATE_PARAM_INPUT_2.text())

		self.MIN_RADIUS_HOLE = int(self.MIN_RADIUS_HOLE_INPUT_2.text())
		self.MAX_RADIUS_HOLE = int(self.MAX_RADIUS_HOLE_INPUT_2.text())
		self.HOLE_DIAMETER = int(self.HOLE_DIAMETER_INPUT_2.text())

		self.DILATE_KERNEL_WIDTH = int(self.DILATE_KERNEL_WIDTH_INPUT_2.text())
		self.DILATE_KERNEL_HEIGHT = int(self.DILATE_KERNEL_HEIGHT_INPUT_2.text())

		self.trialInfoButton.setEnabled(False)
		self.checkTargetBoxButton.setEnabled(False)
		self.specifyTargetBoxButton.setEnabled(False)
		self.confirmAnalysisParametersButton.setEnabled(False)
		self.runAnalysisButton.setEnabled(False)

		self.saveSessionButton.setEnabled(False)

		self.runAnalysesButton.setEnabled(True)

		self.consoleOutputText.append('Video analysis parameters confirmed.\n')

	def selectTarget(self):

		# //
		#    This method loads in the first frame of the video and allows the user to make a target hole selection.
		# //

		self.RUN_MULTIPLE = False

		ret, frame = self.cap.read()
		self.frame_count = self.frame_count + 1

		self.output = frame.copy()

		rows, cols, color = frame.shape
		self.out = cv2.VideoWriter(self.output_filename, self.fourcc, self.fps, (cols, rows))

		# convert video to grayscale
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# apply blur
		frame_blur = cv2.blur(frame_gray, (self.BLUR_KERNEL_WIDTH, self.BLUR_KERNEL_HEIGHT))

		# apply thresholding
		ret_binary, frame_binary = cv2.threshold(frame_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		height, width = frame_binary.shape
		# cv2.imshow('frame_binary', frame_binary)

		# find the table
		frame_floodfill = frame_binary.copy()
		
		h, w = frame_binary.shape[:2]
		mask = np.zeros((h+2, w+2), np.uint8)

		cv2.floodFill(frame_floodfill,mask,(0,0),255)
		
		self.circle_img = cv2.bitwise_not(cv2.bitwise_not(frame_floodfill) | frame_binary)
		self.circle_img = cv2.dilate(self.circle_img, np.ones((self.DILATE_KERNEL_WIDTH, self.DILATE_KERNEL_HEIGHT),np.uint8), iterations=1)

		# cv2.imshow('circle_img', self.circle_img)

		contour_image = cv2.bitwise_not(self.circle_img.copy())
		image, contours, hierarchy = cv2.findContours(contour_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

		# fitting circle to table
		cnt = max(contours, key = cv2.contourArea)
		(x,y), radius = cv2.minEnclosingCircle(cnt)
		center = (int(x), int(y))
		radius = int(radius)
		self.table = Table(center, radius)

		# apply table mask
		self.frame_masked = cv2.bitwise_or(frame_binary, self.circle_img)

		# cv2.imshow('frame_masked', self.frame_masked)

		# segment holes
		contour_image = self.frame_masked.copy()
		image, contours, hierarchy = cv2.findContours(contour_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

		# fitting circles to holes
		self.holes = []
		radii = []
		i = 0
		for cnt in contours:

			(x,y), radius = cv2.minEnclosingCircle(cnt)
			center = (int(x), int(y))
			radius = int(radius)

			if radius > self.MIN_RADIUS_HOLE and radius < self.MAX_RADIUS_HOLE:
				new_hole = Hole(center, radius, i)
				self.holes.append(new_hole)
				radii.append(radius)
				i = i+1

		self.PIXEL_TO_CM = float(self.HOLE_DIAMETER) / (2*stats.mode(radii)[0][0])

		# print len(self.holes)

		# numbering holes
		angles = []
		for hole in self.holes:
			middle_x = int(width/2)
			middle_y = int(height/2)
			angle = np.arctan2(hole.center[1]-middle_y, hole.center[0]-middle_x)
			angles.append(angle - np.pi/2)
		sorted_by_angle = sorted(zip(angles, self.holes), key=lambda x: x[0], reverse=False)

		i = 0
		for element in sorted_by_angle:
			i = i+1
			element[1].number = (i-5)%20 if np.abs(i-5) > 0 else 20

		# draw holes on frame
		for hole in self.holes:
			cv2.putText(self.output, "{}".format(hole.number),
					(hole.center[0]+15, hole.center[1]+15),
					cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
			self.output = cv2.circle(self.output, (hole.center[0], hole.center[1]), hole.radius,(0,255,0),1)

		# draw table on frame
		self.output = cv2.circle(self.output, (self.table.center[0], self.table.center[1]), self.table.radius, (255,255,0), 1)

		cv2.imshow(self.mouse.id, self.output)

		self.specifyTargetBoxButton.setEnabled(True)

	def specifyTargetBox(self):
		# //
		#	 This method confirms the target hole selection.
		# //

		self.TARGET = int(self.specifyTargetBoxLineEdit.text())

		for hole in self.holes:
			if hole.number == self.TARGET:
				hole.isTarget = True
				self.target_hole = hole

		# also initialize the mouse positions
		self.output = self.segmentMouse(self.mouse, self.holes, self.frame_masked, self.output, self.frame_count)
		cv2.imshow(self.mouse.id, self.output)

		# if mouse is at target hole, increment corresponding counter
		if self.target_hole.isActive:
			self.target_hole_activity = self.target_hole_activity + 1
		else:
			self.target_hole_activity = 0

		self.out.write(self.output)
		self.consoleOutputText.append('Target Box Selected: ' + str(self.TARGET) + '\n')
		self.headTailYes.setEnabled(True)
		self.headTailNo.setEnabled(True)

	def setUserSwapNo(self):
		self.USER_SWAP = False
		self.consoleOutputText.append('Head and tail points were not swapped.\n')
		self.runAnalysisButton.setEnabled(True)

	def setUserSwapYes(self):
		self.USER_SWAP = True
		self.consoleOutputText.append('Head and tail points were swapped by user.\n')
		self.runAnalysisButton.setEnabled(True)

	def segmentMouse(self, mouse, holes, input_frame, output_frame, frame_count):

		#1. Find mouse contour.

		# apply mask on holes to isolate mouse
		radii = []
		hole_img = np.ones(input_frame.shape, np.uint8) * 0
		for hole in holes:
			radii.append(hole.radius)
			cv2.circle(hole_img, hole.center, hole.radius+self.HOLE_MASK_FACTOR, 255, thickness=-1)
		input_frame = cv2.bitwise_or(input_frame, hole_img)

		# copy input_frame to find contours on
		contour_image = input_frame.copy()
		image, contours, hierarchy = cv2.findContours(contour_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

		# find contour areas
		areas = []
		for cnt in contours:
			areas.append(cv2.contourArea(cnt))

		# if largest contour can be found (i.e. mouse is present), segment mouse and update center position
		if len(areas) > 1:

			# find largest contour (which is usually the mouse)
			sorted_by_area = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)
			largest_contour = sorted_by_area[1][1]

			# calculate mouse centroid
			M = cv2.moments(largest_contour)
			mouse_x = int(M['m10']/M['m00'])
			mouse_y = int(M['m01']/M['m00'])

			if frame_count == 1: # (first frame, nothing to compare to, so initialize center position)

				mouse.center.append((mouse_x, mouse_y))

			else: # (not the first frame, check to make sure spurious detections are not used)

				distance_from_prev_center = np.sqrt((mouse_x-mouse.center[-1][0])**2 + (mouse_y-mouse.center[-1][1])**2)
				if distance_from_prev_center < self.PREV_DIST_LIM_1 or mouse.center_track_reset > self.RESET_THRESH:
					# update mouse centroid position if jump from previous position is not too big
					#   or if we've lost tracking and need to reset

					# if we do need to reset, we should do a less stringent check before allowing reset
					if distance_from_prev_center < self.PREV_DIST_LIM_2:
						mouse.center.append((mouse_x, mouse_y))
						mouse.center_track_reset = 0
					else:
						mouse.center_track_reset = mouse.center_track_reset + 1
				else:
					mouse.center.append(mouse.center[-1])
					mouse.center_track_reset = mouse.center_track_reset + 1

			# draw mouse on frame
			output_frame = cv2.drawContours(output_frame, largest_contour, -1, (255,0,0), 2)

		else:

			if frame_count == 1:
				self.consoleOutputText.append('mouse not found on first frame ... fatal error! ... exiting.')
				exit()
			else:
				# don't update position
				mouse.center.append(mouse.center[-1])

		# mouse may still be present when largest contour is not there (i.e. mouse is inside a hole, with a bit of the head sticking out)
		# so always look for head and tail!

		# find head and tail using corner detection
		p0 = cv2.goodFeaturesToTrack(input_frame, mask=None, **self.FEATURE_PARAMS)
		p0 = np.array(p0)

		if not(p0.any() == None) and len(p0) > 1: # (both head and tail found)

			head = (p0[0][0][0], p0[0][0][1])
			tail = (p0[1][0][0], p0[1][0][1])

			if frame_count == 1: # (first frame, nothing to compare to, so initialize head and tail position)
				# # ask user if head and tail are selected for properly (no way for software to know)
				# check = raw_input('Head = red, Tail = green. Is this correct [y/n]? ')
				# if check == 'n':
				# 	# don't swap if incorrect
				# 	tmp = head
				# 	head = tail
				# 	tail = tmp
				# 	self.consoleOutputText.append('   head and tail swap (user), ', )
				# else:
				# 	# swap if correct (for some reason, it tends to track the tail as the head the majority of the time)
				# 	self.consoleOutputText.append('no head and tail swap (user), ',)

				mouse.head.append(head)
				mouse.tail.append(tail)

			else: # (not the first frame, check to make sure spurious detections are not used)

				# 1a. SWAP CHECK
				distance_head_to_prev_head = np.sqrt((head[0]-mouse.head[-1][0])**2 + (head[1]-mouse.head[-1][1])**2)
				distance_tail_to_prev_head = np.sqrt((tail[0]-mouse.head[-1][0])**2 + (tail[1]-mouse.head[-1][1])**2)
				if distance_head_to_prev_head > distance_tail_to_prev_head: 
					# (if distance of current head position to previous head position is greater, probably means a swap has occured)
					# so swap back:
					tmp = head
					head = tail
					tail = tmp
					# self.consoleOutputText.append('   head and tail swap (auto), ',)
				else:
					# self.consoleOutputText.append('no head and tail swap (auto), ',)
					pass

				fhead = head
				ftail = tail

				# 2. DISTANCE CHECK
				distance_from_prev_head = np.sqrt((fhead[0]-mouse.head[-1][0])**2 + (fhead[1]-mouse.head[-1][1])**2)
				if distance_from_prev_head > self.PREV_DIST_LIM_1:
					# do not update mouse head position if jump from previous position is too big
					fhead = mouse.head[-1]

				distance_from_prev_tail = np.sqrt((ftail[0]-mouse.tail[-1][0])**2 + (ftail[1]-mouse.tail[-1][1])**2)
				if distance_from_prev_tail > self.PREV_DIST_LIM_1:
					# do not update mouse tail position if jump from previous position is too big
					ftail = mouse.tail[-1]

				# 3. LOST TRACKING OVERRIDE
				if mouse.head_track_reset > self.RESET_THRESH:

					# if we have lost tracking for a while, we override distance check
					fhead = head
					
					# make sure we're not jumping across the table
					distance_from_prev_head = np.sqrt((fhead[0]-mouse.head[-1][0])**2 + (fhead[1]-mouse.head[-1][1])**2)
					if distance_from_prev_head > self.PREV_DIST_LIM_2:
						fhead = mouse.head[-1]

				if mouse.tail_track_reset > self.RESET_THRESH:

					# if we have lost tracking for a while, we override distance check
					ftail = tail

					# make sure we're not jumping across the table
					distance_from_prev_tail = np.sqrt((ftail[0]-mouse.tail[-1][0])**2 + (ftail[1]-mouse.tail[-1][1])**2)
					if distance_from_prev_tail > self.PREV_DIST_LIM_2:
						ftail = mouse.tail[-1]

				# 4. FINAL HEAD <--> TAIL DISTANCE CHECK
				distance_head_to_tail = np.sqrt((fhead[0]-ftail[0])**2 + (fhead[1]-ftail[1])**2)
				if distance_head_to_tail < stats.mode(radii)[0][0]+self.RANGE_FOR_ACTIVATION*stats.mode(radii)[0][0]:
					# too close!
					fhead = mouse.head[-1]
					ftail = mouse.tail[-1]

				# 5. UPDATE COUNTERS AND POSITIONS
				if fhead == mouse.head[-1]:
					mouse.head_track_reset = mouse.head_track_reset + 1
				else:
					mouse.head_track_reset = 0

				if ftail == mouse.tail[-1]:
					mouse.tail_track_reset = mouse.tail_track_reset + 1
				else:
					mouse.tail_track_reset = 0

				# 6. USER SWAP CHECK
				if self.USER_SWAP == True and frame_count == 2:
					tmp = fhead
					fhead = ftail
					ftail = tmp
				
				if frame_count == 2:
					self.consoleOutputText.append('USER_SWAP = ' + str(self.USER_SWAP))

				mouse.head.append(fhead)
				mouse.tail.append(ftail)

		else:

			if frame_count == 1:
				self.consoleOutputText.append('mouse head and tail could not be found on first frame ... fatal error! ... exiting.')
				exit()
			else:
				# don't update head and tail position
				mouse.head.append(mouse.head[-1])
				mouse.tail.append(mouse.tail[-1])

		for i in range(0, self.KALMAN_UPDATE_PARAM):
			# update the Kalman filters with the mouse points, getting the estimates
			estimate_center = mouse.kalfilt_center.step(mouse.center[-1])
			estimate_head = mouse.kalfilt_head.step(mouse.head[-1])
			estimate_tail = mouse.kalfilt_tail.step(mouse.tail[-1])

			# add the estimates to the trajectory
			estimated_center = [int(c) for c in estimate_center]
			mouse.center_k.append(estimated_center)

			estimated_head = [int(h) for h in estimate_head]
			mouse.head_k.append(estimated_head)

			estimated_tail = [int(t) for t in estimate_tail]
			mouse.tail_k.append(estimated_tail)

		# draw mouse points
		output_frame = cv2.circle(output_frame, tuple(estimated_center), 4, (255, 0, 0), -1)
		
		output_frame = cv2.circle(output_frame, tuple(estimated_head), 4, (0, 0, 255), -1)

		output_frame = cv2.circle(output_frame, tuple(estimated_tail), 4, (0, 0, 255), -1)

		# somehow, my tracking algorithm seems to track the tail as the head most of the time (so I've swapped the display here)
		# WTF?????
		output_frame = cv2.putText(output_frame, "H",
						(estimated_tail[0]+10, estimated_tail[1]+10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
		output_frame = cv2.polylines(output_frame, [np.int32(mouse.tail)], isClosed=False, color=(96,96,96))

		output_frame = cv2.putText(output_frame, "T",
							(estimated_head[0]+10, estimated_head[1]+10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

		# self.consoleOutputText.append(mouse.center_track_reset, mouse.head_track_reset, mouse.tail_track_reset,)

		return output_frame

	def runSingleVideoAnalysis(self):

		self.consoleOutputText.append('Running video analysis on ' + self.vid_filename + ' ...\n')

		self.loadVideoButton.setEnabled(False)
		self.trialInfoButton.setEnabled(False)
		self.checkTargetBoxButton.setEnabled(False)
		self.specifyTargetBoxButton.setEnabled(False)
		self.confirmAnalysisParametersButton.setEnabled(False)
		self.runAnalysisButton.setEnabled(False)
		self.headTailYes.setEnabled(False)
		self.headTailNo.setEnabled(False)
		self.saveSessionButton.setEnabled(False)
		self.loadSessionButton.setEnabled(False)
		self.confirmAnalysisParametersButton_2.setEnabled(False)
		self.runAnalysesButton.setEnabled(False)

		while(self.cap.isOpened()):
			ret, frame = self.cap.read()
			self.frame_count = self.frame_count + 1

			if ret == True and (self.frame_count != self.FRAME_LIM):

				if self.target_hole_activity == self.TARGET_HOLE_ACTIVITY:
					self.target_hole_end = True
					self.consoleOutputText.append('Video paused automatically. Software is not sure if mouse entered target hole.')
					self.consoleOutputText.append('  >> Press \'Enter\' to step through the frames. ')
					self.consoleOutputText.append('  >> Press \'q\' to end the video analysis.\n')

				# if self.target_hole_end == True:
				# 	cv2.imshow(self.mouse.id, self.output)

				if self.target_hole_end == True:
					if (cv2.waitKey(0) == ord('q')):
						break

				self.output = frame.copy()

				# we are at all the other frames

				# convert video to grayscale
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

				# apply blur
				frame_blur = cv2.blur(frame_gray, (self.BLUR_KERNEL_WIDTH, self.BLUR_KERNEL_HEIGHT))

				# apply thresholding
				ret_binary, frame_binary = cv2.threshold(frame_blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

				# apply table mask
				frame_masked = cv2.bitwise_or(frame_binary, self.circle_img)

				# draw holes on frame
				for hole in self.holes:
					cv2.putText(self.output, "{}".format(hole.number),
							(hole.center[0]+15, hole.center[1]+15),
							cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
					self.output = cv2.circle(self.output, (hole.center[0], hole.center[1]), hole.radius,(0,255,0),1)

				# segment the mouse
				self.output = self.segmentMouse(self.mouse, self.holes, frame_masked, self.output, self.frame_count)

				# activate nearest hole if within range
				closest_hole, distance = self.mouse.getNearestHole(self.holes, 'head')

				radii = []
				for hole in self.holes:
					radii.append(hole.radius)
				mode_radius = stats.mode(radii)[0][0]

				if distance <= mode_radius+self.RANGE_FOR_ACTIVATION*mode_radius:
					if closest_hole.isTarget:
						self.TARGET_FLAG = True
						closest_hole.setActive(self.TARGET_FLAG)
						self.output = cv2.circle(self.output, (closest_hole.center[0], closest_hole.center[1]), int(mode_radius+self.RANGE_FOR_ACTIVATION*mode_radius),(0,255,0), 2)
						
						# calculate primary path length, primary latency
						if closest_hole.numTimes == 1:
							self.primary_path_length = self.mouse.getPathLength('center', self.PIXEL_TO_CM)
							self.primary_latency = self.frame_count * self.spf

					else:
						closest_hole.setActive(self.TARGET_FLAG)
						self.output = cv2.circle(self.output, (closest_hole.center[0], closest_hole.center[1]), int(mode_radius+self.RANGE_FOR_ACTIVATION*mode_radius),(0,0,255), 2)
				else:
					closest_hole.isActive = False

				for hole in self.holes:
					if hole.isTarget:
						self.target_hole = hole

				if self.target_hole.isActive:
					self.target_hole_activity = self.target_hole_activity + 1
				else:
					self.target_hole_activity = 0
					self.target_hole_end = False

				# draw table on frame
				self.output = cv2.circle(self.output, (self.table.center[0], self.table.center[1]), self.table.radius, (255,255,0), 1)

				cv2.imshow(self.mouse.id, self.output)
				if not(self.RUN_MULTIPLE): cv2.waitKey(1)

				self.out.write(self.output)

			else:
				break

		self.outputResults()

		self.loadVideoButton.setEnabled(True)
		self.saveSessionButton.setEnabled(True)
		self.loadSessionButton.setEnabled(True)
		self.confirmAnalysisParametersButton_2.setEnabled(True)
		self.runAnalysesButton.setEnabled(True)

	def runMultipleVideoAnalysis(self):

		self.RUN_MULTIPLE = True

		for i in range(0,self.loadSessionTable.rowCount()):

			# 1. set input video
			self.vid_filename = str(self.loadSessionTable.item(i,0).text())

			# 2. initialize variables
			self.initializeVariables()

			# 3. confirm trial information
			self.MOUSE_NAME = str(self.loadSessionTable.item(i,1).text())
			self.MONTH = int(self.loadSessionTable.item(i,2).text())
			self.DAY = int(self.loadSessionTable.item(i,3).text())
			self.TRIAL = int(self.loadSessionTable.item(i,4).text())

			self.mouse.id = self.MOUSE_NAME + '_m' + str(self.MONTH) + 'd' + str(self.DAY) + 't' + str(self.TRIAL)
			self.consoleOutputText.append('Mouse ID: ' + self.mouse.id)

			output_directory = 'results/' + self.MOUSE_NAME
			if not os.path.exists(output_directory):
				os.makedirs(output_directory)
			self.consoleOutputText.append('Output Directory: ' + output_directory)

			self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
			self.output_filename = output_directory + '/' + self.mouse.id + '_tracked.avi'
			self.consoleOutputText.append('Output File: ' + self.output_filename + '\n')	

			# 4. confirm any additional analysis parameters
			self.FRAME_LIM = int(self.loadSessionTable.item(i,8).text())

			self.selectTarget()

			# 5. confirm target selection
			self.TARGET = int(self.loadSessionTable.item(i,5).text())

			for hole in self.holes:
				if hole.number == self.TARGET:
					hole.isTarget = True
					self.target_hole = hole

			# also initialize the mouse positions
			self.output = self.segmentMouse(self.mouse, self.holes, self.frame_masked, self.output, self.frame_count)
			cv2.imshow(self.mouse.id, self.output)

			# if mouse is at target hole, increment corresponding counter
			if self.target_hole.isActive:
				self.target_hole_activity = self.target_hole_activity + 1
			else:
				self.target_hole_activity = 0

			self.out.write(self.output)
			self.consoleOutputText.append('Target Box Selected: ' + str(self.TARGET) + '\n')
			self.runAnalysisButton.setEnabled(True)

			# 6. check user swap and run video analysis
			self.USER_SWAP = True if str(self.loadSessionTable.item(i,7).text()) == 'True' else False
			self.runSingleVideoAnalysis()

		self.loadVideoButton.setEnabled(False)

	# ----- METHODS FOR LOADING AND SAVING FILES ----- #

	def setWorkingDirectory(self):
		self.workingDirectory = str(QtWidgets.QFileDialog.getExistingDirectory(self, 'Set Working Directory', os.path.expanduser('~')))
		if self.workingDirectory == '':
			self.workingDirectory = os.path.expanduser('~')
		self.loadVideoButton.setEnabled(True)

	def loadVideo(self):

		# //
		#    This method allows the user to load in the video file of their choice.
		# //

		self.trialInfoButton.setEnabled(False)
		self.checkTargetBoxButton.setEnabled(False)
		self.specifyTargetBoxButton.setEnabled(False)
		self.confirmAnalysisParametersButton.setEnabled(False)
		self.runAnalysisButton.setEnabled(False)

		prev_filename = str(self.loadVideoFilename.text())
		self.vid_filename = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video File', self.workingDirectory, 'Video Files (*.avi)')[0])
		
		if self.vid_filename == '':
			self.vid_filename = prev_filename
			if prev_filename == '':
				self.consoleOutputText.append('Video Loaded From:\n  >> ' + 'Video not loaded. Please try again.\n')
			else:
				self.loadVideoFilename.setText(self.vid_filename)
				self.consoleOutputText.append('Video Loaded From:\n  >> ' + self.vid_filename + '\n')
				self.initializeVariables()
				self.trialInfoButton.setEnabled(True)
		else:
			self.loadVideoFilename.setText(self.vid_filename)
			self.consoleOutputText.append('Video Loaded From:\n  >> ' + self.vid_filename + '\n')
			self.initializeVariables()
			self.trialInfoButton.setEnabled(True)

	def outputResults(self):

		total_primary_errors = 0
		total_secondary_errors = 0

		total_primary_errors_time = 0
		total_secondary_errors_time = 0

		total_target_visits = 0
		total_target_time = 0

		self.consoleOutputText.append('')
		self.consoleOutputText.append('==============================')
		self.consoleOutputText.append('')

		self.output_filename_txt = self.output_filename.replace('avi', 'txt')
		output_txt = open(self.output_filename_txt, 'w')
		print self.output_filename_txt

		self.consoleOutputText.append('results from: ' + str(self.vid_filename))
		self.consoleOutputText.append('mouse id: ' + str(self.mouse.id))
		self.consoleOutputText.append('target: ' + str(self.TARGET))
		self.consoleOutputText.append('')
		output_txt.write('results from:\t' + str(self.vid_filename) + '\n')
		output_txt.write('mouse id:\t' + str(self.mouse.id) + '\n')
		output_txt.write('target:\t' + str(self.TARGET) + '\n')
		output_txt.write('\n')

		self.consoleOutputText.append('hole_number number_of_times_visited primary_errors secondary_errors time_spent_at_hole target_hole')
		output_txt.write('hole_number\tnumber_of_times_visited\tprimary_errors\tsecondary_errors\ttime_spent_at_hole\ttarget_hole\n')
		for hole in self.holes:
			self.consoleOutputText.append(str(hole.number)+' '+str(hole.numTimes)+' '+str(hole.numTimesPrimary)+' '+str(hole.numTimesSecondary)+' '+str(hole.numFrames*self.spf)+' '+str(hole.isTarget))
			output_txt.write(str(hole.number)+'\t'+str(hole.numTimes)+'\t'+str(hole.numTimesPrimary)+'\t'+str(hole.numTimesSecondary)+'\t'+str(hole.numFrames*self.spf)+'\t'+str(hole.isTarget)+'\n')
			
			total_primary_errors = total_primary_errors + hole.numTimesPrimary
			total_primary_errors_time = total_primary_errors_time + hole.numFramesPrimary

			total_secondary_errors = total_secondary_errors + hole.numTimesSecondary
			total_secondary_errors_time = total_secondary_errors_time + hole.numFramesSecondary

			if hole.isTarget:
				total_target_visits = total_target_visits + hole.numTimes
				total_target_time = total_target_time + hole.numFrames

		self.consoleOutputText.append('')
		self.consoleOutputText.append('total_primary_errors total_secondary_errors total_primary_errors_time total_secondary_errors_time total_target_visits total_target_time')
		self.consoleOutputText.append(str(total_primary_errors)+' '+str(total_secondary_errors)+' '+str(total_primary_errors_time*self.spf)+' '+str(total_secondary_errors_time*self.spf)+' '+str(total_target_visits)+' '+str(total_target_time*self.spf))
		output_txt.write('\n')
		output_txt.write('total_primary_errors\ttotal_secondary_errors\ttotal_primary_errors_time\ttotal_secondary_errors_time\ttotal_target_visits\ttotal_target_time\n')
		output_txt.write(str(total_primary_errors)+'\t'+str(total_secondary_errors)+'\t'+str(total_primary_errors_time*self.spf)+'\t'+str(total_secondary_errors_time*self.spf)+'\t'+str(total_target_visits)+'\t'+str(total_target_time*self.spf)+'\n')

		self.consoleOutputText.append('')
		self.consoleOutputText.append('primary_path_length total_path_length')
		if self.primary_path_length == 0: self.primary_path_length = self.mouse.getPathLength('center', self.PIXEL_TO_CM)
		self.consoleOutputText.append(str(self.primary_path_length)+' '+str(self.mouse.getPathLength('center', self.PIXEL_TO_CM)))
		output_txt.write('\n')
		output_txt.write('primary_path_length\ttotal_path_length\n')
		output_txt.write(str(self.primary_path_length)+'\t'+str(self.mouse.getPathLength('center', self.PIXEL_TO_CM))+'\n')

		self.consoleOutputText.append('')
		self.consoleOutputText.append('primary_latency total_latency')
		if self.primary_latency == 0: self.primary_latency = self.frame_count*self.spf
		self.consoleOutputText.append(str(self.primary_latency)+' '+str(self.frame_count*self.spf))
		output_txt.write('\n')
		output_txt.write('primary_latency\ttotal_latency\n')
		output_txt.write(str(self.primary_latency)+'\t'+str(self.frame_count*self.spf) + '\n')

		self.consoleOutputText.append('')
		self.consoleOutputText.append('proximity_to_target')
		self.consoleOutputText.append(str(np.mean(self.mouse.getProxToTarget('center', self.target_hole, self.PIXEL_TO_CM))))
		output_txt.write('\n')
		output_txt.write('proximity_to_target\n')
		output_txt.write(str(np.mean(self.mouse.getProxToTarget('center', self.target_hole, self.PIXEL_TO_CM))) + '\n')


		self.consoleOutputText.append('')
		self.consoleOutputText.append('Video: ' + self.vid_filename)
		self.consoleOutputText.append('FPS: ' + str(self.fps))
		self.consoleOutputText.append('Number of Frames: ' +  str(self.frame_count) + ' ' + str(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
		self.consoleOutputText.append('Length of Video: ' + str(self.frame_count*self.spf) + ' ' + str(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)*(1/self.fps)))
		output_txt.write('\n')
		output_txt.write('Video:\t' + self.vid_filename + '\n')
		output_txt.write('FPS:\t' + str(self.fps) + '\n')
		output_txt.write('Number of Frames:\t' + str(self.frame_count) + '\t' + str(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) + '\n')
		output_txt.write('Length of Video:\t' + str(self.frame_count*self.spf) + '\t' + str(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)*(1/self.fps)) + '\n')

		self.consoleOutputText.append('')
		self.consoleOutputText.append('target_hole_end ' + str(self.target_hole_end))
		output_txt.write('\ntarget_hole_end = ' + str(self.target_hole_end) + '\n')

		self.consoleOutputText.append('')
		self.consoleOutputText.append('ending frame ' + str(self.frame_count))
		output_txt.write('ending frame = ' + str(self.frame_count) + '\n')

		self.consoleOutputText.append('')
		self.consoleOutputText.append('USER_SWAP ' + str(self.USER_SWAP))
		output_txt.write('USER_SWAP = ' + str(self.USER_SWAP) + '\n')

		self.consoleOutputText.append('')
		self.consoleOutputText.append('Results output to ' + self.output_filename + ' ' + self.output_filename_txt)

		self.consoleOutputText.append('')
		self.consoleOutputText.append('==============================')
		self.consoleOutputText.append('')

		self.addToSession()

		# Write information for post-hoc registering of table and paths.
		
		#  | whole path
		output_path_txt = open(self.output_filename_txt.replace('.txt', '_path.txt'), 'w')
		path = self.mouse.getPath('center')
		for i in range(1,len(path)):
			output_path_txt.write(str(path[i][0]) + ',' + str(path[i][1]) + '\n')
		output_path_txt.close()
		
		#  | target location
		output_target_txt = open(self.output_filename_txt.replace('.txt', '_target.txt'), 'w')
		output_target_txt.write(str(self.target_hole.center[0]) + ',' + str(self.target_hole.center[1]) + '\n')
		output_target_txt.write(str(self.target_hole.radius) + '\n')
		output_target_txt.write(str(self.PIXEL_TO_CM))
		output_target_txt.close()

		#  | table location
		output_table_txt = open(self.output_filename_txt.replace('.txt', '_table.txt'), 'w')
		output_table_txt.write(str(self.table.center[0]) + ',' + str(self.table.center[1]) + '\n')
		output_table_txt.write(str(self.table.radius) + '\n')
		output_table_txt.write(str(self.PIXEL_TO_CM))
		output_table_txt.close()

		self.cap.release()
		self.out.release()
		output_txt.close()
		cv2.destroyAllWindows()

	def addToSession(self):
		rowPosition = self.saveSessionTable.rowCount()
		self.saveSessionTable.insertRow(rowPosition)

		self.saveSessionTable.setItem(rowPosition, 0, QtWidgets.QTableWidgetItem(str(self.vid_filename)))
		self.saveSessionTable.setItem(rowPosition, 1, QtWidgets.QTableWidgetItem(str(self.MOUSE_NAME)))
		self.saveSessionTable.setItem(rowPosition, 2, QtWidgets.QTableWidgetItem(str(self.MONTH)))
		self.saveSessionTable.setItem(rowPosition, 3, QtWidgets.QTableWidgetItem(str(self.DAY)))
		self.saveSessionTable.setItem(rowPosition, 4, QtWidgets.QTableWidgetItem(str(self.TRIAL)))
		self.saveSessionTable.setItem(rowPosition, 5, QtWidgets.QTableWidgetItem(str(self.TARGET)))
		self.saveSessionTable.setItem(rowPosition, 6, QtWidgets.QTableWidgetItem(str(self.target_hole_end)))
		self.saveSessionTable.setItem(rowPosition, 7, QtWidgets.QTableWidgetItem(str(self.USER_SWAP)))
		self.saveSessionTable.setItem(rowPosition, 8, QtWidgets.QTableWidgetItem(str(self.frame_count)))
		self.saveSessionTable.setItem(rowPosition, 9, QtWidgets.QTableWidgetItem(str(self.output_filename)))
		self.saveSessionTable.setItem(rowPosition, 10, QtWidgets.QTableWidgetItem(str(self.output_filename_txt)))

	def saveSession(self):
		session_output_filename = 'sessions/BMATS_session_' + datetime.datetime.now().strftime("%Y_%m_%d")
		session_output_filename = self.incrementFilename(session_output_filename, 0, 'sesh')

		with open(session_output_filename, "w") as session_output_file:
			for i in range(0, self.saveSessionTable.rowCount()):
				session_output_file.write(str(self.saveSessionTable.item(i, 0).text()) + '\t')
				session_output_file.write(str(self.saveSessionTable.item(i, 1).text()) + '\t')
				session_output_file.write(str(self.saveSessionTable.item(i, 2).text()) + '\t')
				session_output_file.write(str(self.saveSessionTable.item(i, 3).text()) + '\t')
				session_output_file.write(str(self.saveSessionTable.item(i, 4).text()) + '\t')
				session_output_file.write(str(self.saveSessionTable.item(i, 5).text()) + '\t')
				session_output_file.write(str(self.saveSessionTable.item(i, 6).text()) + '\t')
				session_output_file.write(str(self.saveSessionTable.item(i, 7).text()) + '\t')
				session_output_file.write(str(self.saveSessionTable.item(i, 8).text()) + '\t')
				session_output_file.write(str(self.saveSessionTable.item(i, 9).text()) + '\t')
				session_output_file.write(str(self.saveSessionTable.item(i, 10).text()) + '\t')
				session_output_file.write('\n')

		self.consoleOutputText.append('Session file wrote to: ' + session_output_filename + '\n')

	def incrementFilename(self, filename, num, extension):
		
		filename = filename.replace('.'+extension,'').replace('_('+str(num-1)+')','')

		if num == 0:
			filename = filename + '.' + extension
		else:
			filename = filename + '_(' + str(num) + ').' + extension
		
		if os.path.isfile(filename):
			num = num + 1
			filename = self.incrementFilename(filename, num, extension)

		return filename

	def loadSessions(self):
		sesh_filenames = QtWidgets.QFileDialog.getOpenFileNames(self, 'Open Session File(s)', 'sessions', 'Session Files (*.sesh)')[0]
		
		if len(sesh_filenames) > 0:

			for sesh_filename in sesh_filenames:
				
				self.consoleOutputText.append('Session Loaded From:\n  >> ' + sesh_filename + '\n')

				sesh_file = open(sesh_filename, "r")
				for line in sesh_file:
					rowPosition = self.loadSessionTable.rowCount()
					self.loadSessionTable.insertRow(rowPosition)

					self.loadSessionTable.setItem(rowPosition, 0, QtWidgets.QTableWidgetItem(str(line.split('\t')[0])))
					self.loadSessionTable.setItem(rowPosition, 1, QtWidgets.QTableWidgetItem(str(line.split('\t')[1])))
					self.loadSessionTable.setItem(rowPosition, 2, QtWidgets.QTableWidgetItem(str(line.split('\t')[2])))
					self.loadSessionTable.setItem(rowPosition, 3, QtWidgets.QTableWidgetItem(str(line.split('\t')[3])))
					self.loadSessionTable.setItem(rowPosition, 4, QtWidgets.QTableWidgetItem(str(line.split('\t')[4])))
					self.loadSessionTable.setItem(rowPosition, 5, QtWidgets.QTableWidgetItem(str(line.split('\t')[5])))
					self.loadSessionTable.setItem(rowPosition, 6, QtWidgets.QTableWidgetItem(str(line.split('\t')[6])))
					self.loadSessionTable.setItem(rowPosition, 7, QtWidgets.QTableWidgetItem(str(line.split('\t')[7])))
					self.loadSessionTable.setItem(rowPosition, 8, QtWidgets.QTableWidgetItem(str(line.split('\t')[8])))
					self.loadSessionTable.setItem(rowPosition, 9, QtWidgets.QTableWidgetItem(str(line.split('\t')[9])))
					self.loadSessionTable.setItem(rowPosition, 10, QtWidgets.QTableWidgetItem(str(line.split('\t')[10])))
				sesh_file.close()

			self.confirmAnalysisParametersButton_2.setEnabled(True)

	def summarizeResults(self):

		output_filename = 'results/BMATS_results_summary_' + datetime.datetime.now().strftime("%Y_%m_%d")
		output_filename = self.incrementFilename(output_filename, 0, 'csv')
		outfile = open(output_filename, 'w')

		heading = 'Mouse ID,Time Point,Day,Target,+1,+2,+3,+4,+5,+6,+7,+8,+9,Opposite,-9,-8,-7,-6,-5,-4,-3,-2,-1,Target,+1,+2,+3,+4,+5,+6,+7,+8,+9,Opposite,-9,-8,-7,-6,-5,-4,-3,-2,-1,Average Primary Latency,Average Total Latency,Average Primary Errors,Average Primary Error Time,Average Secondary Errors,Average Secondary Error Time,Average Total Errors,Average Total Error Time,Average Target Visit,Average Target Visit Time,Primary Path Length,Total Path Length,Proximity To Target\n'
		outfile.write(heading)

		for summarize_id in self.summarize_ids:

			if os.path.exists('results/' + summarize_id):

				for (i, summarize_day) in enumerate(self.summarize_days):

					self.consoleOutputText.append('')
					self.consoleOutputText.append('==========')
					self.consoleOutputText.append('')

					poke_dist = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
					time_dist = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
					
					primary_latency = 0
					total_latency = 0
					
					primary_errors = 0
					secondary_errors = 0
					
					primary_error_time = 0
					secondary_error_time = 0
					
					target_visits = 0
					target_time = 0
					
					total_errors = 0
					total_error_time = 0

					primary_path_length = 0
					total_path_length = 0

					prox_to_target = 0

					for trial in range(1,int(self.summarize_trials[i])+1):

						infilename = 'results/' + summarize_id + '/' + summarize_id + '_m' + self.summarize_timepoint + 'd' + str(summarize_day) + 't' + str(trial) + '_tracked.txt'
						infile = open(infilename, 'r')

						self.consoleOutputText.append('')
						self.consoleOutputText.append(infilename)
						print infilename
						self.consoleOutputText.append('Mouse: ' + summarize_id)
						self.consoleOutputText.append('Time Point: ' + self.summarize_timepoint)
						self.consoleOutputText.append('Day: ' + str(summarize_day))
						self.consoleOutputText.append('Trial: ' + str(trial))

						for (ini, inline) in enumerate(infile):

							if ini == 2:
								# self.consoleOutputText.append(inline)
								# print inline
								target = int(inline.split('\t')[1])
								self.consoleOutputText.append('\nTarget: ' + str(target))

							elif ini >= 5 and ini <= 24:
								# self.consoleOutputText.append(inline)
								# print inline
								hole_info = inline.split('\t')

								hole_num = int(hole_info[0])
								hole_num = (hole_num - 20-target)%20
								if hole_num > 10: hole_num = hole_num%10-10

								hole_pokes = int(hole_info[1])
								hole_time = float(hole_info[4])

								poke_dist[hole_num] = poke_dist[hole_num] + hole_pokes
								time_dist[hole_num] = time_dist[hole_num] + hole_time

								self.consoleOutputText.append('\nHole: ' + str(hole_num) + ' ; Pokes: ' + str(hole_pokes))
								self.consoleOutputText.append('Hole: ' + str(hole_num) + ' ; Time: ' + str(hole_time))

							elif ini == 27:
								# self.consoleOutputText.append(inline)
								# print inline
								primary_errors = primary_errors + int(inline.split('\t')[0])
								secondary_errors = secondary_errors + int(inline.split('\t')[1])
								total_errors = total_errors + int(inline.split('\t')[0]) + int(inline.split('\t')[1])

								primary_error_time = primary_error_time + float(inline.split('\t')[2])
								secondary_error_time = secondary_error_time + float(inline.split('\t')[3])
								total_error_time = total_error_time + float(inline.split('\t')[2]) + float(inline.split('\t')[3])

								target_visits = target_visits + float(inline.split('\t')[4])
								target_time = target_time + float(inline.split('\t')[5])

							elif ini == 30:
								primary_path_length = primary_path_length + float(inline.split('\t')[0])
								total_path_length = total_path_length + float(inline.split('\t')[1])

							elif ini == 33:
								# self.consoleOutputText.append(inline)
								# print inline
								primary_latency = primary_latency + float(inline.split('\t')[0])
								total_latency = total_latency + float(inline.split('\t')[1])
								# print 'Primary Latency:', primary_latency
							elif ini == 36:
								prox_to_target = prox_to_target + float(inline.replace('\n', ''))

						infile.close()

					for (hole, count) in enumerate(poke_dist):
						poke_dist[hole] = float(count) / int(self.summarize_trials[i])

					for (hole, time) in enumerate(time_dist):
						time_dist[hole] = float(time) / int(self.summarize_trials[i])

					primary_errors = float(primary_errors) / int(self.summarize_trials[i])
					secondary_errors = float(secondary_errors) / int(self.summarize_trials[i])

					primary_error_time = float(primary_error_time) / int(self.summarize_trials[i])
					secondary_error_time = float(secondary_error_time) / int(self.summarize_trials[i])

					total_errors = float(total_errors) / int(self.summarize_trials[i])
					total_error_time = float(total_error_time) / int(self.summarize_trials[i])

					target_visits = float(target_visits) / int(self.summarize_trials[i])
					target_time = float(target_time) / int(self.summarize_trials[i])

					primary_path_length = float(primary_path_length) / int(self.summarize_trials[i])
					total_path_length = float(total_path_length) / int(self.summarize_trials[i])

					primary_latency = float(primary_latency) / int(self.summarize_trials[i])
					total_latency = float(total_latency) / int(self.summarize_trials[i])

					prox_to_target = float(prox_to_target) / int(self.summarize_trials[i])

					outfile.write(str(summarize_id) +','+ str(self.summarize_timepoint) +','+ str(summarize_day) +','+ str(poke_dist).replace(' ','').replace('[','').replace(']','') +','+ str(time_dist).replace(' ','').replace('[','').replace(']','') +','+ str(primary_latency) +','+ str(total_latency) +','+ str(primary_errors) +','+ str(primary_error_time) +','+ str(secondary_errors) +','+ str(secondary_error_time) +','+ str(total_errors) +','+ str(total_error_time)  +','+ str(target_visits) +','+ str(target_time) +','+ str(primary_path_length) +','+ str(total_path_length) + ',' + str(prox_to_target) + '\n')
					self.consoleOutputText.append(str(summarize_id) +','+ str(self.summarize_timepoint) +','+ str(summarize_day) +','+ str(poke_dist).replace(' ','').replace('[','').replace(']','') +','+ str(time_dist).replace(' ','').replace('[','').replace(']','') +','+ str(primary_latency) +','+ str(total_latency) +','+ str(primary_errors) +','+ str(primary_error_time) +','+ str(secondary_errors) +','+ str(secondary_error_time) +','+ str(total_errors) +','+ str(total_error_time)  +','+ str(target_visits) +','+ str(target_time) +','+ str(primary_path_length) +','+ str(total_path_length) + ',' + str(prox_to_target) + '\n')
					self.consoleOutputText.append('')
					self.consoleOutputText.append('==========')
			else:
				print summarize_id, 'does not exist.'

		outfile.close()

# launch application
if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	window = MyApp()
	window.show()
	sys.exit(app.exec_())
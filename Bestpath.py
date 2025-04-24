import heapq

class EvacuationPathfinder:
    def _init_(self, graph, crowd_density, obstacles, exit_points):
        self.graph = graph  # خريطة المكان كـ Graph
        self.crowd_density = crowd_density  # كثافة الأشخاص عند كل نقطة
        self.obstacles = obstacles  # قائمة بالعوائق (مثلاً نقاط الحريق)
        self.exit_points = exit_points  # قائمة بالمخارج

    def heuristic(self, node, goal):
        """حساب المسافة التقديرية (Manhattan distance) للمخرج الأقرب"""
        x1, y1 = node
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)

    def get_neighbors(self, node):
        """إرجاع الجيران الممكنين مع التحقق من العوائق"""
        neighbors = self.graph.get(node, {})
        return {n: w for n, w in neighbors.items() if n not in self.obstacles}

    def calculate_weight(self, current, neighbor):
        """حساب الوزن الفعلي للطريق بناءً على العوامل المختلفة"""
        base_weight = self.graph[current][neighbor]  # المسافة الأساسية
        crowd_factor = 1 + self.crowd_density.get(neighbor, 0) * 2  # تأثير الازدحام
        return base_weight * crowd_factor  # الوزن المعدّل

    def find_best_exit(self, start):
        """إيجاد أفضل مخرج بناءً على A*"""
        pq = []
        heapq.heappush(pq, (0, start))
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        previous_nodes = {node: None for node in self.graph}

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_node in self.exit_points:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                return path[::-1]  # إرجاع المسار بالعكس (من البداية إلى النهاية)

            for neighbor in self.get_neighbors(current_node):
                weight = self.calculate_weight(current_node, neighbor)
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    estimated_total = distance + min(self.heuristic(neighbor, exit) for exit in self.exit_points)
                    heapq.heappush(pq, (estimated_total, neighbor))

        return None  # إذا لم يكن هناك مسار آمن

# 🏟 مثال على المكان (خريطة تفاعلية)
graph = {
    (0, 0): {(0, 1): 1, (1, 0): 1},
    (0, 1): {(0, 0): 1, (0, 2): 1, (1, 1): 1},
    (0, 2): {(0, 1): 1, (1, 2): 1},
    (1, 0): {(0, 0): 1, (1, 1): 1},
    (1, 1): {(1, 0): 1, (0, 1): 1, (1, 2): 1, (2, 1): 1},
    (1, 2): {(1, 1): 1, (0, 2): 1, (2, 2): 1},
    (2, 1): {(1, 1): 1, (2, 2): 1},
    (2, 2): {(2, 1): 1, (1, 2): 1}
}

# 📌 كثافة الحشود عند كل نقطة (قيم عشوائية بين 0 و 1)
crowd_density = {
    (0, 1): 0.3,
    (1, 1): 0.7,
    (1, 2): 0.2,
    (2, 1): 0.5
}

# 🔥 مواقع العوائق (مثلاً مواقع الحريق)
obstacles = [(1, 1)]

# 🚪 مواقع المخارج
exit_points = [(2, 2)]

# 🔍 تشغيل الخوارزمية من نقطة بداية (0,0)
evacuation = EvacuationPathfinder(graph, crowd_density, obstacles, exit_points)
best_path = evacuation.find_best_exit((0, 0))

print("🚀 best EXIT:", best_path)